#pip install sentence-transformers -q per i modelli locali
#pip install openai -q per API openai
#pip install scikit-learn -q per calcolare la similarità coseno

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

print("libreries imported successfully")


sentences = [
    "The cat is on the table.",
    "The cat is in the garden.",
    "tomorrow is a sunny day.",
]

print("sentences defined successfully")

for i,i2 in enumerate(sentences):
    print(f"Sentence {i}: {i2}")

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("model loaded successfully")

if torch.cuda.is_available():
    model = model.to('cuda')
    print("model moved to GPU successfully")
else:
    print("GPU not available, using CPU")
st_embeddings = model.encode(sentences)
print("embeddings generated successfully")
print(f"Embeddings shape: {st_embeddings.shape}")
print(f"First embedding vector: {st_embeddings[0][:5]}...")  # Print the first 5 values of the first embedding vector
print("---Similarità coseno sentence Transformers---")
if "st_embeddings" in locals() and isinstance(st_embeddings, np.ndarray):
    similarity_matrix_st = cosine_similarity(st_embeddings)
    print("\nSimilarità coseno (Sentence Transformers):")
    print(np.round(similarity_matrix_st, decimals=4))

    print(f"\ninterpretazione:")
    print(f"Frase 0: {sentences[0]}")
    print(f"Frase 1: {sentences[1]}")
    print(f"Frase 2: {sentences[2]}")

    print(f"Similarità tra frase 0 e frase 1: {similarity_matrix_st[0][1]:.4f}")
    print(f"Similarità tra frase 0 e frase 2: {similarity_matrix_st[0][2]:.4f}")
    print(f"Similarità tra frase 1 e frase 2: {similarity_matrix_st[1][2]:.4f}")
else:
    print("Errore: st_embeddings non è definito o non è un array numpy. Non è possibile calcolare la similarità coseno.")