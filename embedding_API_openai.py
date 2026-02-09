import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from openai import OpenAI
import torch
from dotenv import load_dotenv

load_dotenv()  # carica .env dalla root del progetto

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Errore: variabile OPENAI_API_KEY non trovata.")
else:
    print("Chiave API trovata")
    try:
        client = OpenAI(api_key=api_key)
        print("Invio richiesta all'API di OpenAI...")
        print("Chiave recuperata correttamente.")
    except Exception as e:
        print("\nSi è verificato un errore durante la chiamata all'API di OpenAI:")
        print(repr(e))  
sentences = [
    "The cat is on the table.",
    "The dog is in the garden.",
    "tomorrow is a sunny day.",
]
openai_embeddings_np = None

if client:
    try:
        openai_model_name = "text-embedding-ada-002"
        print(f"Richiedendo embedding con il modello {openai_model_name}...")
        response = client.embeddings.create(
            model=openai_model_name,
            input= sentences
        )
        openai_embeddings_list = [item.embedding for item in response.data]
        openai_embeddings_np = np.array(openai_embeddings_list)
        print("Embeddings generati con successo.")
        print(f"Embeddings shape: {openai_embeddings_np.shape}")
        print(f"First embedding vector: {openai_embeddings_np[0][:5]}...")
    except Exception as e:
        print("\nSi è verificato un errore durante la generazione degli embedding con OpenAI:")
        print(repr(e))
    except openai.APIError as e:
        print("\nSi è verificato un errore API durante la generazione degli embedding con OpenAI:")
        print(repr(e))
else:
    print("Client OpenAI non disponibile, non è stato possibile generare gli embedding.")