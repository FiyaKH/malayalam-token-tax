"""README figures: token tax, tokens per word vs vocabulary size, and the swap test.
The numbers are the final results of measure_tax.py, train_tokenizers.py and swap_rules.py.
Writes results/fig_token_tax.png, results/fig_vocab_size.png, results/fig_swap_test.png.
"""
try:                                   # office network: use the Windows certificate store
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
GREY, BLUE, RED, GREEN = "#9aa5b1", "#2b6cb0", "#c53030", "#2f855a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---------- 1. Token tax of six tokenizers ----------
tax = [("MuRIL", 1.18), ("XLM-RoBERTa", 1.38), ("BLOOM", 1.39), ("GPT-4o", 1.95), ("Qwen2.5", 7.23), ("GPT-4", 8.94)]
fig, ax = plt.subplots(figsize=(8, 3.8))
names, vals = zip(*tax)
cols = [RED if v > 5 else BLUE for v in vals]
ax.barh(names, vals, color=cols, height=0.6)
ax.axvline(1, color="black", lw=1, ls="--")
ax.text(1.08, -0.55, "no tax (same as English)", fontsize=8)
for i, v in enumerate(vals):
    ax.text(v + 0.12, i, f"{v:.2f}×", va="center", fontsize=9)
ax.set_xlabel("Malayalam tokens ÷ English tokens for the same sentences (FLORES+, 2,009 pairs)")
ax.set_xlim(0, 10)
ax.set_title("The Malayalam token tax: same meaning, up to 9× the tokens", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig("results/fig_token_tax.png", dpi=150); plt.close(fig)

# ---------- 2. Tokens per word vs vocabulary size ----------
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot([4, 8, 16, 32], [2.91, 2.49, 2.18, 1.93], "o-", color=BLUE, lw=2, label="My BPE, marks-aware split rule")
ax.plot([16], [7.33], "s", color=RED, ms=9, label="My BPE, GPT-2 split rule (16k)")
ax.plot([16], [2.11], "D", color=GREEN, ms=8, label="My Unigram (16k)")
ax.axhline(16.16, color=GREY, ls=":", lw=1.5); ax.text(4, 16.4, "GPT-4: 16.16 tokens per word (100k vocabulary)", fontsize=8, color="#4a5568")
ax.axhline(2.17, color=GREY, ls="--", lw=1); ax.text(22, 2.4, "MuRIL: 2.17", fontsize=8, color="#4a5568")
ax.annotate("same vocabulary size,\n3.4× fewer tokens", xy=(16, 2.25), xytext=(19, 5.3), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.annotate("", xy=(16, 2.35), xytext=(16, 7.1), arrowprops=dict(arrowstyle="<->", color="black", lw=1))
ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels(["4k", "8k", "16k", "32k"])
ax.set_ylim(0, 18)
ax.set_xlabel("Vocabulary size"); ax.set_ylabel("Malayalam tokens per word")
ax.set_title("Fixing one regular expression matters more than vocabulary size", loc="left", fontweight="bold")
ax.legend(fontsize=8, loc="center right", frameon=False)
fig.tight_layout(); fig.savefig("results/fig_vocab_size.png", dpi=150); plt.close(fig)

# ---------- 3. Swap test: vocabulary vs split rule ----------
fig, ax = plt.subplots(figsize=(8, 3.4))
rows = [("GPT-4 vocabulary + GPT-4 rule", 8.94, GREY), ("GPT-4 vocabulary + GPT-4o rule", 8.94, RED),
        ("GPT-4o vocabulary + GPT-4 rule", 3.33, RED), ("GPT-4o vocabulary + GPT-4o rule", 1.95, GREY)]
for i, (n, v, c) in enumerate(rows[::-1]):
    ax.barh(i, v, color=c, height=0.6)
    ax.text(v + 0.12, i, f"{v:.2f}×", va="center", fontsize=9)
ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlim(0, 10.5)
ax.set_xlabel("Malayalam token tax (red = swapped split rule)")
ax.set_title("Swap test: is the tax in the vocabulary or the split rule?", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig("results/fig_swap_test.png", dpi=150); plt.close(fig)
print("Saved results/fig_token_tax.png, results/fig_vocab_size.png, results/fig_swap_test.png")

# ---------- 4. One sentence, many tokenizers (for the README; printed, not drawn,
#               because matplotlib cannot shape Malayalam script correctly) ----------
try:
    import tiktoken
    from tokenizers import Tokenizer
    s = "കേരളം ഇന്ത്യയുടെ തെക്കേ അറ്റത്തുള്ള ഒരു സംസ്ഥാനമാണ്."   # my own sentence (not from FLORES)
    print("\nSentence:", s, f"({len(s.split())} words)")
    for name in ["cl100k_base", "o200k_base"]:
        print(f"{name}: {len(tiktoken.get_encoding(name).encode(s))} tokens")
    for name in ["bpe_gpt2split_16k", "bpe_fixed_16k", "unigram_16k"]:
        t = Tokenizer.from_file(f"tokenizers/ml_{name}.json")
        ids = t.encode(s).ids
        pieces = [t.decode([i]).strip() for i in ids]
        print(f"{name}: {len(ids)} tokens | " + " · ".join(pieces))
except Exception as e:
    print("Token example skipped:", e)