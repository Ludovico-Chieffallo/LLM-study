###############################################################################
# OBIETTIVO GENERALE DELLO SCRIPT
###############################################################################
# Questo script fa “retrieval” (ricerca) su un piccolo indice vettoriale salvato
# su disco ("simple_index.pkl"), usando embeddings e cosine similarity.
#
# Pipeline completa:
# 1) Carica un indice pickle: lista di (embedding_documento, metadati_documento).
# 2) Carica lo STESSO modello usato per creare l’indice.
# 3) Converte una query testuale in embedding.
# 4) Calcola la cosine similarity tra query_embedding e ogni doc_embedding.
# 5) Ordina per similarità decrescente e mostra i TOP-K risultati.
#
# Concetto chiave:
# - L’indice qui non è FAISS, non è un DB vettoriale: è una lista Python.
# - È O(N): per ogni query confronti con tutti i documenti.
# - Va benissimo per didattica o piccoli dataset.
###############################################################################


###############################################################################
# 1) IMPORT DELLE LIBRERIE
###############################################################################
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- SPIEGAZIONE MINUZIOSA ---
# import os
# - Serve per operazioni “di sistema”, qui: os.getcwd() (stampa directory corrente).

# import pickle
# - Per caricare l'indice salvato in precedenza come file .pkl.

# from pathlib import Path
# - Path è un modo più moderno e robusto di gestire percorsi file rispetto a stringhe.
# - Funziona bene cross-platform (Windows/Mac/Linux).
# - Qui lo userai per costruire il percorso dell'indice in modo affidabile.

# import numpy as np
# - Serve per garantire che embeddings siano array NumPy e per manipolare shape.

# import torch
# - Serve per riconoscere tensori PyTorch e convertirli in NumPy (CPU).

# from sentence_transformers import SentenceTransformer
# - Per caricare il modello encoder e creare l'embedding della query.

# from sklearn.metrics.pairwise import cosine_similarity
# - Funzione pronta di scikit-learn per calcolare cosine similarity tra vettori.
# - Perché usarla invece di implementarla a mano?
#   - È affidabile, testata, gestisce shape 2D correttamente.
#   - Riduce bug su normalizzazione e broadcasting.
# - Nota: richiede input 2D (matrici) tipo (1, D), non (D,).


###############################################################################
# 2) FUNZIONE: load_index
###############################################################################
def load_index(index_path: Path):
    if not index_path.exists():
        raise FileNotFoundError(f"Indice non trovato: {index_path}")

    with index_path.open("rb") as f:
        loaded_index = pickle.load(f)

    if not loaded_index:
        raise ValueError(f"Indice caricato ma vuoto/corrotto: {index_path}")

    first_emb, first_ref = loaded_index[0]
    first_emb = np.asarray(first_emb).reshape(-1)

    if not isinstance(first_ref, dict):
        raise TypeError("Formato indice non valido: il riferimento documento non è un dict.")

    return loaded_index, first_emb.shape[0]

# --- SPIEGAZIONE MINUZIOSA (riga per riga) ---
# def load_index(index_path: Path):
# - Definisce una funzione che carica l’indice da disco.
# - index_path è tipizzato come Path: indica che ci aspettiamo un oggetto Path.
# - Nota: il type hint non “impone” a runtime, ma aiuta editor/linters/lettura.

# if not index_path.exists():
# - Verifica se il file esiste.
# - Perché farlo PRIMA del load?
#   - Per dare un errore più chiaro e controllato.

# raise FileNotFoundError(...)
# - Solleva (non stampa) un’eccezione specifica: comportamento corretto per funzioni.
# - Così chi chiama la funzione può gestire l’errore (try/except nel main).

# with index_path.open("rb") as f:
# - Apre il file in lettura binaria (rb).
# - È obbligatorio per pickle, che legge bytes.
# - with garantisce chiusura file automatica.

# loaded_index = pickle.load(f)
# - Carica l’oggetto serializzato.
# - Qui ci aspettiamo: lista di tuple (embedding, reference_dict).

# if not loaded_index:
# - Se loaded_index è:
#   - [] (lista vuota) => False
#   - None => False
#   - altra struttura vuota => False
# - Quindi blocchi il caso “indice vuoto o corrotto”.

# raise ValueError(...)
# - ValueError è appropriato quando il file c’è ma il contenuto non è valido.

# first_emb, first_ref = loaded_index[0]
# - Prendi la prima voce come “campione” per validare la struttura.
# - Se loaded_index è una lista di (embedding, reference), questa riga funziona.
# - Se non lo è, qui potrebbe esplodere con errori (TypeError/IndexError).

# first_emb = np.asarray(first_emb).reshape(-1)
# - np.asarray converte first_emb in un array NumPy (se non lo è già).
# - .reshape(-1) “appiattisce” a vettore 1D:
#   - se first_emb era (1, D) -> diventa (D,)
#   - se era già (D,) resta (D,)
# - Perché lo fai?
#   - Per ottenere la dimensione D in modo uniforme con shape[0].

# if not isinstance(first_ref, dict):
# - Verifica che il “riferimento documento” sia un dizionario.
# - Questo ti protegge da file pickle “sbagliati” o cambiamenti di formato.

# raise TypeError(...)
# - TypeError è sensato perché il “tipo” del riferimento non è quello atteso.

# return loaded_index, first_emb.shape[0]
# - Ritorni:
#   1) l’indice completo (lista).
#   2) doc_dim = dimensione dell’embedding del documento (D).
# - Perché restituire doc_dim?
#   - Serve per controllare compatibilità con l’embedding della query (deve avere stesso D).


###############################################################################
# 3) FUNZIONE: ensure_numpy_embedding
###############################################################################
def ensure_numpy_embedding(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

# --- SPIEGAZIONE MINUZIOSA ---
# def ensure_numpy_embedding(x) -> np.ndarray:
# - Funzione di utility: prende un embedding e garantisce che ritorni np.ndarray.
# - Può arrivare come:
#   - np.ndarray
#   - list
#   - torch.Tensor
#   - anche array-like di vario tipo

# if torch.is_tensor(x):
# - Controlla se x è un tensore PyTorch.

# x = x.detach().cpu().numpy()
# - detach():
#   - stacca il tensore dal grafo di autograd (non vuoi gradienti qui).
# - cpu():
#   - sposta su CPU (necessario perché .numpy() su tensori GPU non è consentito).
# - numpy():
#   - converte a NumPy array.

# return np.asarray(x)
# - Converte comunque a np.ndarray:
#   - se era già NumPy, è quasi “no-op” (o crea view).
#   - se era lista, la trasforma in array.
# - Perché non usare np.array invece di np.asarray?
#   - np.array tende a copiare più spesso.
#   - np.asarray prova a evitare copie inutili (più efficiente).


###############################################################################
# 4) FUNZIONE PRINCIPALE: main
###############################################################################
def main():
    base_dir = Path(__file__).resolve().parent
    index_filepath = base_dir / "simple_index.pkl"

    print("Working directory:", os.getcwd())
    print("Indice atteso in:", index_filepath)

    try:
        loaded_index, doc_dim = load_index(index_filepath)
        print(f"Indice caricato: {len(loaded_index)} documenti. Dimensione embedding doc: {doc_dim}")
    except Exception as e:
        print(f"ERRORE: {e}")
        return

    model_name_used_for_index = "paraphrase-multilingual-mpnet-base-v2"

    try:
        print(f"Caricamento modello: {model_name_used_for_index} ...")
        model = SentenceTransformer(model_name_used_for_index)
        print("Modello caricato.")
    except Exception as e:
        print(f"ERRORE caricamento modello: {e}")
        return

    query = "Quali animali sono considerati i migliori amici dell'uomo?"
    print("Query:", query)

    try:
        query_embedding = model.encode([query])
        query_embedding = ensure_numpy_embedding(query_embedding).reshape(1, -1)
        print("Shape embedding query:", query_embedding.shape)
    except Exception as e:
        print(f"ERRORE generazione embedding query: {e}")
        return

    query_dim = query_embedding.shape[1]
    if query_dim != doc_dim:
        print(
            f"ERRORE: dimensioni embedding non compatibili.\n"
            f"- Query dim: {query_dim}\n"
            f"- Doc dim:   {doc_dim}\n"
            f"Probabile causa: l'indice è stato creato con un modello diverso da '{model_name_used_for_index}'."
        )
        return

    all_similarities = []

    for doc_embedding, doc_reference in loaded_index:
        doc_embedding = ensure_numpy_embedding(doc_embedding).reshape(1, -1)

        score = float(cosine_similarity(query_embedding, doc_embedding)[0][0])
        all_similarities.append((score, doc_reference))

    all_similarities.sort(key=lambda x: x[0], reverse=True)

    top_k = 3
    print(f"\nTop {top_k} risultati:")
    for i, (score, ref) in enumerate(all_similarities[:top_k], start=1):
        doc_id = ref.get("id", "N/A")
        text = ref.get("text", ref.get("preview", ""))
        print(f"\n{i}. Similarità: {score:.4f}")
        print(f"   ID: {doc_id}")
        print(f"   Testo: {text}")

    if len(all_similarities) > top_k:
        score_last, ref_last = all_similarities[-1]
        print("\nEsempio documento meno simile:")
        print(f"   Similarità: {score_last:.4f}")
        print(f"   ID: {ref_last.get('id', 'N/A')}")
        print(f"   Testo: {ref_last.get('text', ref_last.get('preview', ''))}")

# --- SPIEGAZIONE MINUZIOSA (a blocchi e riga per riga) ---

# def main():
# - Punto d’ingresso logico dello script.
# - Buona pratica: separare la logica in una funzione e non “sparare” tutto a livello globale.
# - Permette import di questo file come modulo senza eseguire subito il codice.

# base_dir = Path(__file__).resolve().parent
# - __file__ è il percorso del file Python corrente.
# - Path(__file__) lo trasforma in oggetto Path.
# - .resolve() lo rende assoluto e risolve eventuali symlink.
# - .parent prende la cartella che contiene questo script.
# Perché è importante?
# - Ti permette di trovare l’indice RELATIVAMENTE allo script, non alla directory di lancio.
# - Evita bug classici: “funziona in IDE ma non da terminale”.

# index_filepath = base_dir / "simple_index.pkl"
# - Operatore / in pathlib: unisce path in modo pulito.
# - Esempio: base_dir=".../progetto" -> ".../progetto/simple_index.pkl"

# print("Working directory:", os.getcwd())
# - Mostra la directory corrente del processo.
# - Utile per debug: spesso i file non si trovano per colpa del cwd.

# print("Indice atteso in:", index_filepath)
# - Mostra dove si aspetta di trovare il file.

# try:
#   loaded_index, doc_dim = load_index(index_filepath)
# - Chiama la funzione di caricamento.
# - doc_dim è D, la dimensione embedding dei documenti.

# except Exception as e:
# - Cattura QUALSIASI errore (generico).
# - Pro: non crasha, stampa messaggio e termina.
# - Contro: troppo generico, in progetti seri catturi errori specifici.

# return
# - Esce dalla funzione main: termina lo script in modo pulito.

# model_name_used_for_index = "paraphrase-multilingual-mpnet-base-v2"
# - IMPORTANTISSIMO: qui dichiari il modello che PRESUMI sia stato usato per creare l’indice.
# - Perché serve?
#   - Per avere embeddings compatibili (stessa dimensione e “spazio semantico”).

# model = SentenceTransformer(model_name_used_for_index)
# - Carica il modello per encodare la query.

# query = "Quali animali sono considerati i migliori amici dell'uomo?"
# - La query utente da cercare tra i documenti.
# - Nota: documenti sono inglesi, query italiana.
#   Se il modello è multilingua, può comunque funzionare decentemente.
#   Con un modello solo inglese, sarebbe molto peggio.

# query_embedding = model.encode([query])
# - encode vuole una lista di frasi.
# - Anche per una sola query, passi [query] (lista di 1 elemento).
# - Questo produce tipicamente una matrice shape (1, D).
#   (dipende, ma spesso sì)

# query_embedding = ensure_numpy_embedding(query_embedding).reshape(1, -1)
# - ensure_numpy_embedding: garantisce che sia np.ndarray.
# - reshape(1, -1):
#   - forza la forma 2D (1, D).
#   - -1 significa “calcola automaticamente D”.
# Perché serve?
# - cosine_similarity si aspetta input 2D: (n_samples, n_features).
# - Una query singola è “1 campione”.

# query_dim = query_embedding.shape[1]
# - Se query_embedding è (1, D):
#   - shape[0] = 1 (numero campioni)
#   - shape[1] = D (features) => dimensione embedding.

# if query_dim != doc_dim:
# - Controllo fondamentale:
#   - Se l’indice è stato creato con un altro modello, potresti avere D diverso.
#   - Anche se D fosse uguale, lo “spazio semantico” potrebbe essere diverso:
#     ma qui almeno controlli la compatibilità dimensionale.
# - Quando fallisce, stampi un messaggio molto esplicativo e interrompi.

# all_similarities = []
# - Lista dove accumulerai (score, reference) per ogni documento.

# for doc_embedding, doc_reference in loaded_index:
# - loaded_index è una lista di tuple: (embedding, dict).
# - Iteri su ogni documento e prendi:
#   - doc_embedding: vettore del documento
#   - doc_reference: metadati (id, text, preview, ecc.)

# doc_embedding = ensure_numpy_embedding(doc_embedding).reshape(1, -1)
# - Stessa logica della query:
#   - converti in NumPy
#   - forzi shape (1, D) per cosine_similarity.

# score = float(cosine_similarity(query_embedding, doc_embedding)[0][0])
# - cosine_similarity(A, B) con A=(1,D) e B=(1,D) restituisce matrice (1,1).
# - [0][0] prende lo scalare.
# - float(...) converte in float Python puro (più comodo da stampare/serializzare).
#
# Cosine similarity in parole semplici:
# - misura quanto due vettori “puntano nella stessa direzione”.
# - range tipico: [-1, 1]
#   - 1: identici come direzione (molto simili)
#   - 0: ortogonali (non correlati)
#   - -1: opposti (raro in embedding moderni, ma possibile)

# all_similarities.append((score, doc_reference))
# - Salvi lo score e il riferimento.
# - NON salvi l'embedding (non serve più dopo il calcolo).
# - Scelta efficiente: riduci memoria.

# all_similarities.sort(key=lambda x: x[0], reverse=True)
# - Ordini in base allo score (x[0]) decrescente.
# - lambda x: x[0] è una funzione anonima che estrae il primo elemento della tupla.
# - reverse=True => dal più simile al meno simile.

# top_k = 3
# - Quanti risultati vuoi mostrare.

# for i, (score, ref) in enumerate(all_similarities[:top_k], start=1):
# - all_similarities[:top_k] prende i primi K.
# - enumerate(..., start=1) numerazione umana (1,2,3...) invece di (0,1,2...).
# - (score, ref) spacchetta la tupla.

# doc_id = ref.get("id", "N/A")
# - ref è un dict.
# - .get("id", "N/A") prende id se esiste, altrimenti "N/A".
# - Perché usare get?
#   - Evita KeyError se il dict non ha la chiave.

# text = ref.get("text", ref.get("preview", ""))
# - Preferisci mostrare il testo completo "text".
# - Se manca, provi con "preview".
# - Se manca anche "preview", stringa vuota.
# - È una gestione robusta.

# print(f"\n{i}. Similarità: {score:.4f}")
# - :.4f formatta con 4 decimali, output più leggibile.

# if len(all_similarities) > top_k:
# - Se hai più documenti di quelli mostrati, fai vedere un esempio “peggiore”.

# score_last, ref_last = all_similarities[-1]
# - [-1] prende l’ultimo elemento: il meno simile (dato l’ordinamento decrescente).

# ref_last.get('text', ref_last.get('preview', ''))
# - Stessa logica robusta di prima.


###############################################################################
# 5) ENTRYPOINT STANDARD PYTHON
###############################################################################
if __name__ == "__main__":
    main()

# --- SPIEGAZIONE MINUZIOSA ---
# if __name__ == "__main__":
# - __name__ vale "__main__" solo quando esegui questo file direttamente:
#   python script.py
# - Se invece importi questo file da un altro:
#   import script
#   allora __name__ sarà "script" (o nome modulo), e main() NON viene eseguito.
#
# Perché è importante?
# - Ti permette di riusare funzioni (load_index, ensure_numpy_embedding) senza far partire
#   l’esecuzione automaticamente.
# - È una best practice standard Python.
###############################################################################
