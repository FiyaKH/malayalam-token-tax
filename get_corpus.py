import truststore
truststore.inject_into_ssl()                          # office network certificates

import os
from datasets import load_dataset

# Malayalam Wikipedia (Wikimedia dump of 1 Nov 2023), streamed so we can stop at a size limit
ds = load_dataset("wikimedia/wikipedia", "20231101.ml", split="train", streaming=True)

LIMIT = 60_000_000                                    # ~60 million characters is plenty to train a tokenizer
os.makedirs("data", exist_ok=True)
n_articles, n_chars = 0, 0
with open("data/ml_wiki.txt", "w", encoding="utf-8") as f:
    for row in ds:
        text = row["text"].strip()
        if len(text) < 200:                           # skip stubs
            continue
        f.write(text + "\n")
        n_articles += 1
        n_chars += len(text)
        if n_articles % 5000 == 0:
            print(f"{n_articles} articles, {n_chars / 1e6:.1f} M characters")
        if n_chars >= LIMIT:
            break
print(f"Done: {n_articles} articles, {n_chars / 1e6:.1f} M characters saved to data/ml_wiki.txt")