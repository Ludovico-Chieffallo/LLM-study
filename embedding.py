#pip install sentence-transformers -q per i modelli locali
#pip install openai -q per API openai
#pip install scikit-learn -q per calcolare la similarità coseno

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from sentence_transformers import SentenceTransformer
import openai
import torch

print("libreries imported successfully")


sentences = [
    "The cat is on the table.",
    "The dog is in the garden.",
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
