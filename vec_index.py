import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pickle
import os

print("libreries imported successfully")

#corpus di esempio
documents = [
    "the cat is a small animal.",
    "dogs are famous for their loyalty.",
    "the lifecycle of a butterfly includes the stages of egg, larva, pupa, and adult.",
    "the sun is a star at the center of our solar system.",
    "the human brain is a complex organ responsible for thought and emotion."
    "the history of the Roman Empire is a fascinating subject of study.",
    "the process of photosynthesis allows plants to convert sunlight into energy.",
    "the theory of relativity revolutionized our understanding of space and time.",
    "the Great Wall of China is a remarkable feat of engineering.",
    "the Amazon rainforest is home to a diverse array of plant and animal species."
]
print("documents defined successfully")
print(f"Number of documents: {len(documents)}")
for i, doc in enumerate(documents):
    print(f"Document {i}: {doc[:50]}...")  # Print the first 50 characters of each document
# Load the pre-trained model
model_name_to_load = "paraphrase-multilingual-mpnet-base-v2"

corpus_model = None
current_model_name_in_memory = None
if "active_corpus_model" in globals() and "active_model_name" in globals():
    if active_model_name == model_name_to_load:
        corpus_model = active_corpus_model
        current_model_name_in_memory = active_model_name
    else:
        print(f"Il modello attivo in memoria ({active_model_name}) è diverso da quello richiesto ({model_name_to_load}). Caricamento del nuovo modello...")
if corpus_model is None:
    print(f"Caricamento del modello '{model_name_to_load}'...")
    try:
        from sentence_transformers import SentenceTransformer
        corpus_model = SentenceTransformer(model_name_to_load)
        current_model_name_in_memory = model_name_to_load
        print(f"Modello '{model_name_to_load}' caricato con successo.")

        globals()['active_corpus_model'] = corpus_model
        globals()['active_model_name'] = current_model_name_in_memory
        print(f"Modello '{model_name_to_load}' memorizzato in variabili globali.")
    except Exception as e:
        print(f"Si è verificato un errore durante il caricamento del modello '{model_name_to_load}': {repr(e)}")
    except ImportError as e:
        print(f"Si è verificato un errore di importazione durante il caricamento del modello '{model_name_to_load}': {repr(e)}")


if corpus_model:
    if torch.cuda.is_available():
        try:
            device = next(corpus_model.parameters()).device
            if "cuda" in str(device):
                print("Il modello è già sulla GPU.")
            else:
                corpus_model = corpus_model.to('cuda')
                print("Modello spostato sulla GPU con successo.")
        except Exception as e:
            print(f"Si è verificato un errore durante lo spostamento del modello sulla GPU: {repr(e)}")
    else:
        print("GPU non disponibile, utilizzando CPU.")
else:
    print("Il modello non è stato caricato correttamente, non è possibile spostarlo sulla GPU.")

corpus_embeddings = None
if corpus_model:
    #modifica la riga per usare current_model_name_in_memory
    print (f"\nGenerazione embeddings per i{len(documents)} documenti usando il modello '{current_model_name_in_memory}'...")
    #show progress bar è utile per documenti più grandi.
    corpus_embeddings = corpus_model.encode(documents, show_progress_bar=True)

    print("Embeddings generati con successo.")
    print(f"Shape degli embeddings: {corpus_embeddings.shape}")

    if corpus_embeddings is not None:
        print(f"Dimensione di un singolo embedding: {corpus_embeddings[0].shape[0]}")
        print(f"Primi 3 valori del primo embedding: {corpus_embeddings[0][:3]}...")
else:
    print("Il modello non è stato caricato correttamente, non è possibile generare gli embeddings.")


simple_index = []
if corpus_embeddings is not None:
    for i, embedding in enumerate(corpus_embeddings):
        doc_reference = {
            "id": i,
            "text": documents[i],
            "preview": documents[i] if len(documents[i]) <= 80 else documents[i][:80] + "..."
        }
        simple_index.append((embedding, doc_reference))

    print(f"Simple index creato con successo. Numero di voci nell'indice: {len(simple_index)}")

    if simple_index:
        print("Struttura di una voce dell'indice:")
        first_item_embedding, first_item_reference = simple_index[0]
        print(f"  - Tipo di embedding: {type(first_item_embedding)}")
        print(f"  - Dimensione dell'embedding: {first_item_embedding.shape}")
        print(f"  - Contenuto del riferimento: {first_item_reference}")
        print(f"  - Testo di anteprima: {first_item_reference['preview']}")
        print(f"  - Primi 3 valori dell'embedding: {first_item_embedding[:3]}...")
else:
    print("Gli embeddings non sono stati generati correttamente, non è possibile creare l'indice.")
