from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
testo = "Ciao, come stai?"
tokens = tokenizer.tokenize(testo)
input_ids = tokenizer.encode(testo)

print("testo originale:", testo)
print("Tokens (subwords):", tokens)
print("Token IDs:", input_ids)