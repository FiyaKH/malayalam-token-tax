"""Step 6: summarise all language-model runs into one table and one chart.
Reads results/lm_*.json, writes results/lm_summary.csv and results/lm_bpc.png.
"""
import glob, json, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in sorted(glob.glob("results/lm_*.json"))]
for r in runs:                                   # the first runs were saved before "run"/"seed" were recorded
    r.setdefault("run", r["tokenizer"]); r.setdefault("seed", 0)

# which runs belong to which bar (same tokenizer, same training budget)
groups = [
    ("BPE, GPT-2 split\nequal tokens", lambda r: r["tokenizer"] == "bpe_gpt2split_16k" and not r["run"].endswith("eqchars")),
    ("BPE, GPT-2 split\nequal characters", lambda r: r["run"].endswith("eqchars")),
    ("BPE, marks-aware split\nequal tokens", lambda r: r["tokenizer"] == "bpe_fixed_16k"),
    ("Unigram\nequal tokens", lambda r: r["tokenizer"] == "unigram_16k"),
]

os.makedirs("results", exist_ok=True)
with open("results/lm_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["setting", "run", "seed", "steps", "training_chars_seen_M", "test_bits_per_char"])
    for name, pick in groups:
        for r in runs:
            if pick(r):
                w.writerow([name.replace("\n", ", "), r["run"], r["seed"], r["steps"],
                            round(r["training_chars_seen"] / 1e6, 1), r["test_bits_per_char"]])

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#b0b0b0", "#7f7f7f", "#2b6cb0", "#2f855a"]
for i, (name, pick) in enumerate(groups):
    vals = [r["test_bits_per_char"] for r in runs if pick(r)]
    if not vals:
        continue
    chars = [r["training_chars_seen"] / 1e6 for r in runs if pick(r)]
    mean = sum(vals) / len(vals)
    ax.bar(i, mean, color=colors[i], width=0.6)
    ax.scatter([i] * len(vals), vals, color="black", s=18, zorder=3)          # each seed as a dot
    seeds = f"{len(vals)} seed" + ("s" if len(vals) > 1 else "")
    ax.text(i, mean + 0.02, f"{mean:.3f}\n{seeds}, {sum(chars) / len(chars):.0f}M chars seen",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylabel("FLORES+ Malayalam bits per character (lower is better)")
ax.set_ylim(1.5, 2.2)
ax.set_title("Tiny Malayalam GPT (2.9M parameters): same model, different tokenizer", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/lm_bpc.png", dpi=150)
print(open("results/lm_summary.csv", encoding="utf-8").read())
print("Saved results/lm_summary.csv and results/lm_bpc.png")