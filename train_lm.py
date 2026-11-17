"""Step 5: does a better tokenizer give a better Malayalam language model?
Trains a tiny GPT-style model on Malayalam Wikipedia with one of our tokenizers, and measures
bits per character (bpc) on the unseen FLORES+ Malayalam sentences. bpc is fair across tokenizers:
it measures how well the model predicts the TEXT, however the text is cut into tokens.

  python train_lm.py --tok bpe_gpt2split_16k     # BPE, GPT-2 split rule (words chopped at vowel signs)
  python train_lm.py --tok bpe_fixed_16k         # BPE, fixed split rule
  python train_lm.py --tok unigram_16k           # Unigram
Control runs get their own folder with --tag, e.g.
  python train_lm.py --tok bpe_gpt2split_16k --steps 18100 --tag bpe_gpt2split_16k_eqchars
  python train_lm.py --tok bpe_fixed_16k --seed 1 --tag bpe_fixed_16k_seed1
Ctrl+C saves a checkpoint; run the same command again to continue.
"""
import truststore
truststore.inject_into_ssl()

import argparse, json, math, os, time, unicodedata
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--tok", required=True, choices=["bpe_gpt2split_16k", "bpe_fixed_16k", "unigram_16k"])
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--ctx", type=int, default=128)          # tokens of context
p.add_argument("--batch", type=int, default=16)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--device", default="cpu", choices=["cpu", "xpu", "cuda"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="", help="name for this run's output folder and results file (default: the tokenizer name)")
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
RUN = args.tag or args.tok
OUT = f"out_lm_{RUN}"; os.makedirs(OUT, exist_ok=True); os.makedirs("results", exist_ok=True)
CKPT = f"{OUT}/checkpoint.pt"
tok = Tokenizer.from_file(f"tokenizers/ml_{args.tok}.json")
V = tok.get_vocab_size()

# ---------- 1. Training data: Malayalam Wikipedia, tokenized once and cached ----------
cache = f"data/ids_{args.tok}.npy"
if not os.path.exists(cache):
    print("Tokenizing the Wikipedia corpus (once)...")
    lines = [unicodedata.normalize("NFC", l.strip()) for l in open("data/ml_wiki.txt", encoding="utf-8") if l.strip()]
    ids = []
    for i in range(0, len(lines), 1000):
        for e in tok.encode_batch([l + " " for l in lines[i:i + 1000]]):
            ids.extend(e.ids)
    np.save(cache, np.array(ids, dtype=np.uint16))
data = np.load(cache)
corpus_chars = sum(len(l.strip()) + 1 for l in open("data/ml_wiki.txt", encoding="utf-8") if l.strip())
print(f"Training corpus: {len(data):,} tokens for {corpus_chars:,} characters ({corpus_chars / len(data):.2f} chars per token)")

# ---------- 2. Test data: FLORES+ Malayalam (never used for training) ----------
from datasets import load_dataset
ds = load_dataset("openlanguagedata/flores_plus", "mal_Mlym")
test_text = " ".join(unicodedata.normalize("NFC", row["text"]) for split in ds for row in ds[split])
test_ids = torch.tensor(tok.encode(test_text).ids)

# ---------- 3. A tiny GPT ----------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)        # each token sees only the past
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(V, args.dim), nn.Embedding(args.ctx, args.dim)
        self.blocks = nn.ModuleList([Block(args.dim, args.heads) for _ in range(args.layers)])
        self.ln = nn.LayerNorm(args.dim)
    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln(x) @ self.tok_emb.weight.T                         # output layer shares the embedding

def init_weights(m):
    # GPT-2 style: small starting weights (std 0.02). PyTorch's default embedding std is 1.0, which with the
    # shared output layer makes the first predictions wildly over-confident and slows learning a lot.
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.zeros_(m.bias)

dev = args.device
model = GPT()
model.apply(init_weights)
for b in model.blocks:                                                   # residual outputs start smaller (GPT-2)
    nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * args.layers))
    nn.init.normal_(b.mlp[2].weight, std=0.02 / math.sqrt(2 * args.layers))
model = model.to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
lr_at = lambda s: args.lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
start = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    print(f"Resumed from step {start}")

@torch.no_grad()
def bits_per_char(ids, text_chars, stride=None):
    """Total prediction cost of the test text, in bits, divided by its number of characters."""
    model.eval()
    stride = stride or args.ctx // 2
    nll, counted = 0.0, 0
    for i in range(0, len(ids) - 1, stride):
        x = ids[i:i + args.ctx]; y = ids[i + 1:i + args.ctx + 1]
        if len(y) < 1: break
        x = x[:len(y)]
        logits = model(x[None].to(dev))[0]
        loss = F.cross_entropy(logits, y.to(dev), reduction="none")
        keep = 0 if i == 0 else args.ctx - stride                         # count each token only once
        nll += loss[keep:].sum().item(); counted += len(loss) - keep
    model.train()
    return nll / math.log(2) / text_chars, counted

# ---------- 4. Training ----------
if start < args.steps:
    print(f"{RUN} | seed {args.seed} | vocab {V} | {n_params:,} parameters | {args.batch} x {args.ctx} tokens per step | "
          f"steps {start} -> {args.steps} | Ctrl+C saves")
    t0, s = time.time(), start
    try:
        for s in range(start, args.steps):
            for g in opt.param_groups: g["lr"] = lr_at(s)
            off = np.random.randint(0, len(data) - args.ctx - 1, args.batch)
            xb = torch.tensor(np.stack([data[o:o + args.ctx] for o in off]).astype(np.int64)).to(dev)
            yb = torch.tensor(np.stack([data[o + 1:o + args.ctx + 1] for o in off]).astype(np.int64)).to(dev)
            loss = F.cross_entropy(model(xb).reshape(-1, V), yb.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if (s + 1) % 250 == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s + 1}, CKPT)
                el = time.time() - t0; done = s + 1 - start
                print(f"step {s + 1:5d} | train loss {loss.item():.3f} | {el / done:.2f} s/step | "
                      f"~{el / done * (args.steps - s - 1) / 60:.0f} min left | saved")
        s = args.steps
    except KeyboardInterrupt:
        print("Stopping: saving checkpoint...")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s}, CKPT)
    start = s

# ---------- 5. Evaluation on FLORES+ (unseen) ----------
bpc, n_tok = bits_per_char(test_ids, len(test_text))
chars_seen = start * args.batch * args.ctx * corpus_chars / len(data)
res = {"run": RUN, "tokenizer": args.tok, "seed": args.seed, "steps": start, "parameters": n_params, "test_bits_per_char": round(bpc, 4),
       "test_tokens": len(test_ids), "test_chars": len(test_text),
       "training_tokens_seen": start * args.batch * args.ctx, "training_chars_seen": int(chars_seen)}
json.dump(res, open(f"results/lm_{RUN}.json", "w"), indent=1)
print(f"\nFLORES+ Malayalam: {bpc:.3f} bits per character (lower is better) | "
      f"model saw ~{chars_seen / 1e6:.1f} M characters of training text")

# ---------- 6. A short sample of generated Malayalam ----------
@torch.no_grad()
def generate(prompt, n=60, temp=0.8):
    model.eval()
    ids = tok.encode(prompt).ids
    for _ in range(n):
        x = torch.tensor(ids[-args.ctx:])[None].to(dev)
        probs = F.softmax(model(x)[0, -1] / temp, -1)
        ids.append(torch.multinomial(probs, 1).item())
    return tok.decode(ids)
sample = generate("കേരളം")
open(f"{OUT}/sample.txt", "w", encoding="utf-8").write(sample)
print(f"Sample saved to {OUT}/sample.txt (open it in Notepad: PowerShell may not show Malayalam correctly)")