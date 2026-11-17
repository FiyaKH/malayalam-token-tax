import truststore
truststore.inject_into_ssl()          # use Windows certificates (office network)

EN = "Kerala is a state on the southwestern coast of India."
ML = "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്തുള്ള ഒരു സംസ്ഥാനമാണ്."

results = []

# 1. OpenAI tokenizers (used by GPT-4 and GPT-4o)
try:
    import tiktoken
    for name in ["cl100k_base", "o200k_base"]:
        enc = tiktoken.get_encoding(name)
        results.append((f"OpenAI {name}", len(enc.encode(EN)), len(enc.encode(ML))))
except Exception as e:
    print("OpenAI tokenizers failed:", type(e).__name__, str(e)[:120])

# 2. Open models on Hugging Face
from transformers import AutoTokenizer
for model in ["Qwen/Qwen2.5-0.5B",            # Chinese-English focused LLM
              "bigscience/bloom-560m",        # multilingual LLM (includes Malayalam in training)
              "xlm-roberta-base",             # multilingual (100 languages)
              "google/muril-base-cased"]:     # built for Indian languages
    try:
        tok = AutoTokenizer.from_pretrained(model)
        n_en = len(tok.encode(EN, add_special_tokens=False))
        n_ml = len(tok.encode(ML, add_special_tokens=False))
        results.append((model, n_en, n_ml))
    except Exception as e:
        print(f"{model} failed:", type(e).__name__, str(e)[:120])

print(f"\n{'tokenizer':32s} {'English':>8s} {'Malayalam':>10s} {'ratio':>7s}")
for name, n_en, n_ml in results:
    print(f"{name:32s} {n_en:8d} {n_ml:10d} {n_ml / n_en:7.2f}x")