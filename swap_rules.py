import truststore
truststore.inject_into_ssl()

import tiktoken
from datasets import load_dataset

# FLORES+ test sentences (English and Malayalam, same meaning)
def load_lang(code):
    ds = load_dataset("openlanguagedata/flores_plus", code)
    return [row["text"] for split in ds for row in ds[split]]
EN, ML = load_lang("eng_Latn"), load_lang("mal_Mlym")

gpt4, gpt4o = tiktoken.get_encoding("cl100k_base"), tiktoken.get_encoding("o200k_base")

def swap(vocab_from, rule_from, name):
    """Same vocabulary (merge list) as `vocab_from`, but the split rule of `rule_from`."""
    return tiktoken.Encoding(name=name, pat_str=rule_from._pat_str,
                             mergeable_ranks=vocab_from._mergeable_ranks, special_tokens={})

variants = [
    ("GPT-4 vocabulary + GPT-4 rule (original)",   gpt4),
    ("GPT-4 vocabulary + GPT-4o rule (marks kept)", swap(gpt4, gpt4o, "gpt4_vocab_gpt4o_rule")),
    ("GPT-4o vocabulary + GPT-4o rule (original)", gpt4o),
    ("GPT-4o vocabulary + GPT-4 rule (marks cut)",  swap(gpt4o, gpt4, "gpt4o_vocab_gpt4_rule")),
]
print("GPT-4o's split rule keeps combining marks (\\p{M}) in words:", "\\p{M}" in gpt4o._pat_str)
print("GPT-4's split rule keeps combining marks (\\p{M}) in words: ", "\\p{M}" in gpt4._pat_str)

ml_words = sum(len(s.split()) for s in ML)
print(f"\n{'variant':46s} {'EN tokens':>10s} {'ML tokens':>10s} {'tax':>6s} {'tok/word':>9s} {'roundtrip':>10s}")
for name, enc in variants:
    n_en = sum(len(enc.encode(s)) for s in EN)
    ids = [enc.encode(s) for s in ML]
    n_ml = sum(len(i) for i in ids)
    ok = sum(enc.decode(i) == s for i, s in zip(ids, ML)) / len(ML)
    print(f"{name:46s} {n_en:10d} {n_ml:10d} {n_ml / n_en:5.2f}x {n_ml / ml_words:9.2f} {ok:10.1%}")