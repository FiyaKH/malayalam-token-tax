import truststore
truststore.inject_into_ssl()

import csv, os, time, unicodedata
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, Regex

CORPUS = "data/ml_wiki.txt"
os.makedirs("tokenizers", exist_ok=True); os.makedirs("results", exist_ok=True)

# Pre-tokenizer rules: text is cut into chunks first, and BPE merges can never cross a chunk boundary.
# GPT-2/GPT-4 style keeps runs of LETTERS (\p{L}) together; Malayalam vowel signs and the virama are
# COMBINING MARKS (\p{M}), so that rule cuts every word at every vowel sign. The fixed rule keeps
# letters AND marks together.
FIXED_SPLIT = r" ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+"

def train_bpe(vocab, fixed_split):
    """Byte-level BPE (as in GPT tokenizers), trained only on Malayalam."""
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    if fixed_split:
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(FIXED_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    else:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)   # GPT-2 split rule
    tok.decoder = decoders.ByteLevel()
    tok.train([CORPUS], trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<unk>"],
                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok

def train_unigram(vocab):
    """Unigram language-model tokenizer (as in SentencePiece), trained only on Malayalam."""
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.train([CORPUS], trainers.UnigramTrainer(vocab_size=vocab, special_tokens=["<unk>"],
                                                unk_token="<unk>", max_piece_length=16))
    return tok

# FLORES+ Malayalam: test set only (never used for training)
ds = load_dataset("openlanguagedata/flores_plus", "mal_Mlym")
ML = [unicodedata.normalize("NFC", row["text"]) for split in ds for row in ds[split]]
words, chars = sum(len(s.split()) for s in ML), sum(len(s) for s in ML)
GPT4_ML_TOKENS = 473735                                  # from measure_tax.py, GPT-4 (cl100k)

configs = [("BPE, GPT-2 split", 16000, lambda v: train_bpe(v, False), "bpe_gpt2split"),
           ("BPE, fixed split", 4000,  lambda v: train_bpe(v, True),  "bpe_fixed"),
           ("BPE, fixed split", 8000,  lambda v: train_bpe(v, True),  "bpe_fixed"),
           ("BPE, fixed split", 16000, lambda v: train_bpe(v, True),  "bpe_fixed"),
           ("BPE, fixed split", 32000, lambda v: train_bpe(v, True),  "bpe_fixed"),
           ("Unigram",          16000, train_unigram,                 "unigram")]
rows = []
for label, vocab, train, fname in configs:
    t0 = time.time()
    tok = train(vocab)
    tok.save(f"tokenizers/ml_{fname}_{vocab // 1000}k.json")
    enc = [tok.encode(s) for s in ML]
    n = sum(len(e.ids) for e in enc)
    roundtrip = sum(tok.decode(e.ids).strip() == s.strip() for e, s in zip(enc, ML)) / len(ML)
    rows.append({"tokenizer": f"Ours: {label} {vocab // 1000}k", "malayalam_tokens": n,
                 "tokens_per_malayalam_word": round(n / words, 2), "malayalam_chars_per_token": round(chars / n, 2),
                 "fewer_tokens_than_gpt4": round(GPT4_ML_TOKENS / n, 1), "exact_roundtrip": round(roundtrip, 3)})
    print(f"trained {label} {vocab // 1000}k in {time.time() - t0:.0f}s")

print(f"\n{'tokenizer':30s} {'ML tokens':>10s} {'tok/word':>9s} {'chars/tok':>10s} {'vs GPT-4':>9s} {'roundtrip':>10s}")
for r in rows:
    print(f"{r['tokenizer']:30s} {r['malayalam_tokens']:10d} {r['tokens_per_malayalam_word']:9.2f} "
          f"{r['malayalam_chars_per_token']:10.2f} {r['fewer_tokens_than_gpt4']:8.1f}x {r['exact_roundtrip']:10.1%}")
with open("results/own_tokenizers.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("Saved results/own_tokenizers.csv and tokenizers/*.json")