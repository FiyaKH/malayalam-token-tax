import truststore
truststore.inject_into_ssl()                          # office network certificates

import csv, os, statistics
from datasets import load_dataset

# ---- 1. Load ~2,000 English-Malayalam sentence pairs with the same meaning (FLORES+ dev + devtest) ----
def load_lang(code):
    ds = load_dataset("openlanguagedata/flores_plus", code)
    return {(split, row["id"]): row["text"] for split in ds for row in ds[split]}   # ids restart in each split

en, ml = load_lang("eng_Latn"), load_lang("mal_Mlym")
ids = sorted(set(en) & set(ml))
EN, ML = [en[i] for i in ids], [ml[i] for i in ids]
print(f"{len(ids)} sentence pairs loaded")

# ---- 2. Tokenizers to compare ----
import tiktoken
from transformers import AutoTokenizer

def openai(name):
    enc = tiktoken.get_encoding(name)
    return lambda s: len(enc.encode(s))

def hf(model):
    tok = AutoTokenizer.from_pretrained(model)
    return lambda s: len(tok.encode(s, add_special_tokens=False))

tokenizers = {
    "OpenAI cl100k (GPT-4)":     openai("cl100k_base"),
    "OpenAI o200k (GPT-4o)":     openai("o200k_base"),
    "Qwen2.5":                   hf("Qwen/Qwen2.5-0.5B"),
    "BLOOM":                     hf("bigscience/bloom-560m"),
    "XLM-RoBERTa":               hf("xlm-roberta-base"),
    "MuRIL (Indian languages)":  hf("google/muril-base-cased"),
}

# ---- 3. Measure ----
ml_words = sum(len(s.split()) for s in ML)
ml_chars = sum(len(s) for s in ML)
rows = []
for name, count in tokenizers.items():
    n_en = [count(s) for s in EN]
    n_ml = [count(s) for s in ML]
    rows.append({
        "tokenizer": name,
        "english_tokens": sum(n_en),
        "malayalam_tokens": sum(n_ml),
        "tax_ratio": round(sum(n_ml) / sum(n_en), 2),                                    # overall
        "median_sentence_ratio": round(statistics.median(m / e for m, e in zip(n_ml, n_en)), 2),
        "tokens_per_malayalam_word": round(sum(n_ml) / ml_words, 2),                     # "fertility"
        "malayalam_chars_per_token": round(ml_chars / sum(n_ml), 2),
    })

print(f"\n{'tokenizer':28s} {'EN tokens':>10s} {'ML tokens':>10s} {'tax':>6s} {'median':>7s} {'tok/word':>9s} {'chars/tok':>10s}")
for r in rows:
    print(f"{r['tokenizer']:28s} {r['english_tokens']:10d} {r['malayalam_tokens']:10d} {r['tax_ratio']:5.2f}x "
          f"{r['median_sentence_ratio']:6.2f}x {r['tokens_per_malayalam_word']:9.2f} {r['malayalam_chars_per_token']:10.2f}")

os.makedirs("results", exist_ok=True)
with open("results/token_tax.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"\nMalayalam: {ml_words} words, {ml_chars} characters. Saved results/token_tax.csv (numbers only, no sentences)")