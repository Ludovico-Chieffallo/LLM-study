###############################################################################
# OBIETTIVO GENERALE DELLO SCRIPT (RAG “classico”)
###############################################################################
# Questo script implementa una pipeline RAG (Retrieval-Augmented Generation):
#
# 1) Retrieval:
#    - Carica un indice di embeddings da file (simple_index.pkl).
#    - Dato un testo query, calcola l’embedding della query.
#    - Calcola cosine similarity tra query e ogni documento indicizzato.
#    - Prende i top-K documenti più simili.
#
# 2) Augmentation:
#    - Costruisce un “prompt” che include:
#      - I documenti top-K (contesto).
#      - Regole rigidissime che obbligano il LLM a rispondere solo col contesto.
#
# 3) Generation:
#    - Chiama OpenAI Chat Completions con il prompt e ottiene una risposta.
#
# NOTA IMPORTANTE:
# - Questo script NON crea l’indice: assume che "simple_index.pkl" esista già.
# - L’indice deve essere compatibile col modello embedding scelto
#   (stesso modello usato per generarlo, altrimenti embedding dimension diversa o spazio diverso).
###############################################################################


###############################################################################
# 0) DIPENDENZE (commentate)
###############################################################################
#pip install sentence-transformers -q
#pip install openai -q
#pip install scikit-learn -q

# --- SPIEGAZIONE MINUZIOSA ---
# Queste righe sono tipiche di notebook (Colab/Jupyter).
# - pip install ... installa librerie mancanti.
# - -q = quiet: meno output.
# In uno script locale “pulito”, di solito:
# - si usano requirements.txt / pyproject.toml
# - e non si mette pip install dentro il codice.


###############################################################################
# 1) IMPORT DELLE LIBRERIE
###############################################################################
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import torch

# --- SPIEGAZIONE MINUZIOSA ---
# import pickle
# - Serve a caricare l’indice salvato con pickle (lista di tuple).

# import os
# - Serve per:
#   - controllare se il file esiste (os.path.exists)
#   - leggere variabili d’ambiente (os.getenv)

# import numpy as np
# - Serve per controllare/conversione degli embeddings (ndarray) e reshape.

# SentenceTransformer
# - Modello embedding per convertire testo -> vettore numerico.

# cosine_similarity
# - Funzione di scikit-learn che calcola la cosine similarity tra matrici 2D.

# import openai
# - Libreria SDK OpenAI per chiamare il LLM.

# import torch
# - Qui usato per:
#   - torch.cuda.is_available() (check GPU)
#   - NOTA: non sposti realmente il modello su GPU in questo script, controlli solo.


###############################################################################
# 2) CARICAMENTO VARIABILI DA .env (OPZIONALE)
###############################################################################
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Se python-dotenv non è installato, va comunque se OPENAI_API_KEY è già nell'ambiente
    pass

print("Librerie importate.")

# --- SPIEGAZIONE MINUZIOSA ---
# try/except:
# - Obiettivo: caricare variabili d'ambiente da un file .env (tipico in sviluppo locale).
#
# from dotenv import load_dotenv
# - dotenv è una libreria esterna ("python-dotenv").
# - Se non è installata, l’import fallisce.
#
# load_dotenv()
# - Legge un file .env nella directory corrente (o in alcuni casi in percorsi standard)
#   e “inietta” coppie KEY=VALUE nelle variabili d’ambiente del processo.
#
# except Exception: pass
# - Se non c’è dotenv, non è un blocco fatale:
#   - lo script può funzionare se OPENAI_API_KEY è già nelle env.
# - Però: catturare Exception è “molto largo”.
#   - Più pulito: except ImportError.
# - Qui va bene per praticità.


###############################################################################
# 3) CARICAMENTO DELL’INDICE (pickle)
###############################################################################
index_filepath = "simple_index.pkl"
loaded_index = None

if os.path.exists(index_filepath):
    try:
        with open(index_filepath, "rb") as f_in:
            loaded_index = pickle.load(f_in)
        if loaded_index:
            print(f"Indice caricato con successo da '{index_filepath}' ({len(loaded_index)} elementi).")
        else:
            print(f"ATTENZIONE: Indice caricato da '{index_filepath}' ma risulta vuoto.")
            loaded_index = None
    except Exception as e:
        print(f"ERRORE durante il caricamento dell'indice da '{index_filepath}': {e}")
else:
    print(f"ERRORE: File indice '{index_filepath}' non trovato. Assicurati di averlo generato/coperto in questa cartella.")

# --- SPIEGAZIONE MINUZIOSA ---
# index_filepath = "simple_index.pkl"
# - Percorso file dell’indice.
# - Qui è relativo alla working directory (cwd), NON al file Python.
#   Quindi se esegui lo script da un'altra cartella, potrebbe non trovarlo.
#   (Nel tuo script precedente usavi Path(__file__).parent: più robusto.)

# loaded_index = None
# - Variabile sentinella: “indice non disponibile”.

# if os.path.exists(index_filepath):
# - Check file esistente, così puoi dare un messaggio chiaro.

# with open(index_filepath, "rb") as f_in:
# - Lettura binaria: necessaria per pickle.

# loaded_index = pickle.load(f_in)
# - Deserializza l’oggetto salvato.
# - Atteso: lista di tuple (embedding, doc_reference_dict).

# if loaded_index:
# - Se la lista non è vuota => ok.
# - Stampa quanti elementi.

# else:
# - Se la lista è vuota, è un segnale di problema:
#   - indice creato male
#   - file corrotto
#   - oppure indice veramente vuoto (caso raro).
# - Imposti loaded_index=None per trattarlo come “non valido”.

# except Exception as e:
# - Qualsiasi errore di caricamento (pickle error, permessi, ecc).
# - Nota sicurezza: pickle non è sicuro se il file viene da fonti non fidate.


###############################################################################
# 4) CARICAMENTO MODELLO EMBEDDING
###############################################################################
model_name_used_for_index = "paraphrase-multilingual-mpnet-base-v2"
embedding_model = None

try:
    print(f"\nCaricamento del modello embedding '{model_name_used_for_index}'...")
    embedding_model = SentenceTransformer(model_name_used_for_index)
    print(f"Modello embedding '{model_name_used_for_index}' caricato con successo.")

    if torch.cuda.is_available():
        print(f"Il modello embedding può usare la GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU non disponibile (uso CPU).")

except Exception as e:
    print(f"ERRORE durante il caricamento del modello embedding '{model_name_used_for_index}': {e}")
    embedding_model = None

# --- SPIEGAZIONE MINUZIOSA ---
# model_name_used_for_index = ...
# - Nome del modello che ASSUMI sia lo stesso con cui hai creato l’indice.
# - È fondamentale per coerenza: stessa dimensione D e stesso spazio semantico.

# embedding_model = None
# - Sentinella: modello non disponibile.

# embedding_model = SentenceTransformer(model_name_used_for_index)
# - Carica modello pre-addestrato (scarica se necessario).
#
# check GPU:
# - torch.cuda.is_available() ti dice se la macchina vede una GPU CUDA.
# - torch.cuda.get_device_name(0) stampa il nome della GPU.
#
# NOTA IMPORTANTISSIMA:
# - Qui tu NON stai spostando esplicitamente il modello su GPU.
# - SentenceTransformer spesso gestisce device automaticamente, ma non sempre.
# - Se vuoi forzare: embedding_model = embedding_model.to("cuda") (se supportato).
# - Quindi la frase “può usare la GPU” è corretta come potenziale,
#   non come certezza che stia già usando la GPU.


###############################################################################
# 5) CONFIGURAZIONE CLIENT OPENAI (SDK MODERNO)
###############################################################################
openai_client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non trovata. "
            "Impostala come variabile d'ambiente o in un file .env (OPENAI_API_KEY=...)."
        )

    # Con SDK OpenAI moderno:
    openai_client = openai.OpenAI(api_key=api_key)

    # Test rapido opzionale (scommenta se vuoi verificare subito):
    # openai_client.models.list()

    print("\nClient OpenAI configurato correttamente (VS Code / ambiente locale).")

except openai.AuthenticationError:
    print("\nERRORE di Autenticazione OpenAI: La API Key fornita non è valida.")
except Exception as e:
    print(f"\nERRORE nella configurazione del client OpenAI: {e}")

# --- SPIEGAZIONE MINUZIOSA ---
# openai_client = None
# - Sentinella: client non configurato.

# api_key = os.getenv("OPENAI_API_KEY")
# - Legge variabile d'ambiente.
# - Se hai fatto load_dotenv e nel .env c’è OPENAI_API_KEY=..., allora qui la trova.

# if not api_key:
# - Se stringa vuota o None, significa che la chiave non è disponibile.

# raise RuntimeError(...)
# - Errore “bloccante” per la pipeline: senza chiave non puoi chiamare il LLM.

# openai_client = openai.OpenAI(api_key=api_key)
# - Crea client OpenAI con API key.
#
# NOTA “di realtà pratica”:
# - La libreria openai ha avuto cambiamenti significativi tra versioni.
# - Questo stile (openai.OpenAI(...)) è tipico delle versioni moderne.
# - Se qualcuno ha una versione vecchia, potrebbe non funzionare.

# except openai.AuthenticationError:
# - Errore specifico: key presente ma invalida o non autorizzata.

# except Exception as e:
# - Qualsiasi altro errore (mancanza rete, SDK mismatch, ecc).


###############################################################################
# 6) CORPUS ORIGINALE (TESTO COMPLETO DEI DOCUMENTI)
###############################################################################
documents_corpus = [
    "Il gatto è un animale domestico popolare.",
    "I cani sono noti per la loro lealtà verso i padroni.",
    "Il ciclo di vita di una farfalla include quattro stadi: uovo, larva, pupa e adulto.",
    "Python è un linguaggio di programmazione versatile e ampiamente utilizzato.",
    "L'intelligenza artificiale sta trasformando molti settori industriali.",
    "La ricetta della torta di mele richiede farina, zucchero, burro e mele.",
    "Il sistema solare è composto da otto pianeti che orbitano attorno al Sole.",
    "Imparare a suonare la chitarra richiede pratica costante."
]
print(f"\nCorpus originale di {len(documents_corpus)} documenti disponibile per il contesto.")

# --- SPIEGAZIONE MINUZIOSA ---
# Questo è il testo “ground truth” dei documenti.
# Perché serve se hai già l’indice?
# - Nel pickle, tu spesso salvi solo un reference con id e magari preview.
# - Qui, usando l’ID, recuperi il testo completo per inserirlo nel prompt.
#
# ATTENZIONE COERENZA:
# - Questo corpus deve essere lo stesso ordine/insieme usato per costruire l’indice!
# - Perché poi fai: doc_id = ref.get("id") e indicizzi documents_corpus[doc_id].
# - Se l’indice era stato creato su documenti diversi/ordine diverso, recuperi testo sbagliato.


###############################################################################
# 7) VERIFICA SETUP (diagnostica)
###############################################################################
print("\n--- VERIFICA SETUP ---")
print("RISULTATO: Indice", "CARICATO." if loaded_index else "NON caricato.")
print("RISULTATO: Modello Embedding", "CARICATO." if embedding_model else "NON caricato.")
print("RISULTATO: Client OpenAI", "CONFIGURATO." if openai_client else "NON configurato.")
print("RISULTATO: Corpus documenti", "DISPONIBILE." if documents_corpus else "NON disponibile.")

if not loaded_index or not embedding_model or not openai_client or not documents_corpus:
    print("\n--- ATTENZIONE: Setup incompleto. La pipeline RAG potrebbe non funzionare correttamente. ---")
else:
    print("\n--- Setup completato con successo! Tutti i componenti per la pipeline RAG sono pronti. ---")

# --- SPIEGAZIONE MINUZIOSA ---
# Queste stampe sono “health checks”:
# - ti dicono subito se un componente critico manca.
# - In pipeline multi-step, questo ti salva tempo di debug.
#
# if not loaded_index or not embedding_model or not openai_client or not documents_corpus:
# - Condizione “fallimento” se qualunque componente è mancante.
# - not documents_corpus sarebbe True solo se la lista fosse vuota [].
#
# Nota: anche se “tutti presenti”, potresti comunque avere problemi di coerenza
# (indice creato con corpus diverso, o modello diverso, ecc).
# Qui controlli solo presenza, non compatibilità totale.


###############################################################################
# 8) FUNZIONE DI RETRIEVAL: search_semantic
###############################################################################
def search_semantic(query: str, index: list, emb_model, top_k: int = 3) -> list:
    if not index or not emb_model:
        print("Errore in search_semantic: Indice o modello di embedding non forniti.")
        return []
    if not query:
        print("Errore in search_semantic: Query vuota.")
        return []

    try:
        query_embedding = emb_model.encode([query])
    except Exception as e:
        print(f"Errore durante la generazione dell'embedding per la query '{query}': {e}")
        return []

    all_similarities = []
    for doc_idx, (doc_embedding_vector, doc_reference) in enumerate(index):
        try:
            if not isinstance(doc_embedding_vector, np.ndarray):
                print(f"Attenzione: embedding non-ndarray per doc {doc_idx}. Salto.")
                continue

            doc_embedding_reshaped = doc_embedding_vector.reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding_reshaped)[0][0]
            all_similarities.append((similarity, doc_reference))
        except Exception as e:
            print(f"Errore nel calcolo similarità (doc {doc_idx}): {e}")

    all_similarities.sort(key=lambda item: item[0], reverse=True)
    return all_similarities[:top_k]

# --- SPIEGAZIONE MINUZIOSA ---
# def search_semantic(...):
# - Input:
#   - query: stringa utente
#   - index: lista di tuple (doc_embedding, doc_reference)
#   - emb_model: SentenceTransformer
#   - top_k: quanti documenti ritornare
# - Output:
#   - lista di tuple (similarity_score, doc_reference) ordinata desc e tagliata a top_k.

# if not index or not emb_model:
# - Evita crash se pipeline non è pronta.

# if not query:
# - Evita query vuota (embedding inutile/errore).

# query_embedding = emb_model.encode([query])
# - Ottieni embedding query.
#
# NOTA TECNICA MOLTO IMPORTANTE:
# - Qui NON converti query_embedding in np.ndarray né fai reshape(1,-1).
# - In molte versioni, encode ritorna già np.ndarray shape (1, D),
#   quindi funziona.
# - Ma se encode tornasse lista o torch.Tensor, potresti avere problemi.
# - Nel tuo script precedente avevi una funzione ensure_numpy_embedding: qui no.

# all_similarities = []
# - Accumula risultati.

# for doc_idx, (doc_embedding_vector, doc_reference) in enumerate(index):
# - enumerate ti dà doc_idx (indice di iterazione) utile per logging.

# if not isinstance(doc_embedding_vector, np.ndarray):
# - Controllo “difensivo”: richiedi che embedding sia np.ndarray.
# - Se nel tuo indice gli embeddings fossero torch.Tensor o list, li scarti.
#   (Questo è un punto critico: potresti perdere tutti i documenti!)
#
# Motivo della scelta:
# - Cosine similarity lavora bene con array 2D numeric.
# - Però: qui sarebbe più robusto convertire invece di saltare.

# doc_embedding_reshaped = doc_embedding_vector.reshape(1, -1)
# - Forza 2D per cosine_similarity:
#   - (D,) -> (1, D)

# similarity = cosine_similarity(query_embedding, doc_embedding_reshaped)[0][0]
# - ritorna matrice (1,1), prendi scalare.

# all_similarities.sort(key=lambda item: item[0], reverse=True)
# - ordina per score decrescente.

# return all_similarities[:top_k]
# - top-k documenti più simili.
#
# COMPLESSITÀ:
# - O(N) confronti, dove N = numero documenti in index.
# - Ok per piccoli dataset, non scala bene per milioni di documenti.


###############################################################################
# 9) COSTRUZIONE DEL PROMPT RAG: build_rag_prompt
###############################################################################
def build_rag_prompt(query: str, search_results: list, original_documents_list: list) -> str:
    context_str = ""
    if not search_results:
        context_str = "Nessun contesto rilevante trovato nei documenti forniti.\n"
    else:
        for i, (score, ref) in enumerate(search_results):
            doc_id = ref.get("id", -1)
            if 0 <= doc_id < len(original_documents_list):
                doc_text = original_documents_list[doc_id]
                context_str += (
                    f"--- Contesto Documento {i+1} (ID: {doc_id}, Similarità Score: {score:.4f}) ---\n"
                    f"{doc_text}\n"
                    "--------------------------------------------------------------------------\n\n"
                )
            else:
                print(f"Attenzione: ID documento {doc_id} non valido. Salto.")

    if not context_str.strip() and search_results:
        context_str = "Contesto rilevante trovato ma impossibile recuperare il testo dei documenti (problema con ID).\n"

    prompt = f"""Istruzioni per l'Assistente AI:
1. Sei un assistente AI che risponde alle domande basandosi ESCLUSIVAMENTE sul "Contesto Fornito" qui sotto.
2. Non usare alcuna conoscenza esterna o informazione pregressa. La tua unica fonte di verità è il Contesto Fornito.
3. Rispondi alla "Domanda Utente" in modo chiaro, conciso e fattuale.
4. Se le informazioni necessarie per rispondere alla Domanda Utente non si trovano nel Contesto Fornito, devi rispondere ESATTAMENTE con la frase: "Le informazioni richieste non sono presenti nei documenti forniti."
5. Non inventare o inferire informazioni che non siano esplicitamente dichiarate nel Contesto Fornito.
6. Se il Contesto Fornito non contiene dati sufficienti per rispondere alla Domanda Utente,
   rispondi esattamente con: "Le informazioni richieste non sono presenti nei documenti forniti."
--- CONTESTO FORNITO ---
{context_str.strip()}
--- FINE CONTESTO ---

--- DOMANDA UTENTE ---
{query}
--- FINE DOMANDA UTENTE ---

Seguendo scrupolosamente TUTTE le Istruzioni sopra, fornisci ora la tua risposta alla Domanda Utente.
RISPOSTA:"""
    return prompt

# --- SPIEGAZIONE MINUZIOSA ---
# build_rag_prompt:
# - Costruisce una grande stringa prompt che contiene:
#   - istruzioni rigide per il modello
#   - contesto recuperato (testo documenti top-k)
#   - domanda utente
#
# context_str = ""
# - stringa che accumula i documenti di contesto.
#
# if not search_results:
# - Se retrieval non trova nulla (o errori), metti una frase di fallback.
#
# else:
# - Per ogni risultato recuperato (score, ref):
#   - score: similarità
#   - ref: dict con id (e magari preview/text).
#
# doc_id = ref.get("id", -1)
# - Recuperi l’ID del documento.
# - Se manca, default -1 (che verrà invalidato dal check dopo).
#
# if 0 <= doc_id < len(original_documents_list):
# - Validazione: evita IndexError.
# - Se id è valido:
#   - doc_text = original_documents_list[doc_id]
#   - aggiungi al contesto.
#
# context_str += (...)
# - Concatenazione stringhe ripetuta.
# - Per pochi documenti va bene.
# - Per molti, sarebbe meglio usare una lista e poi ''.join(lista) (più efficiente).
#
# f"--- Contesto Documento {i+1} ... ---"
# - Aggiungi header con ID e score.
# - Inserire lo score nel prompt può aiutare trasparenza, ma:
#   - non sempre aiuta il LLM a “ragionare meglio”.
#   - però è utile a te per debug.
#
# if not context_str.strip() and search_results:
# - Caso particolare: avevi risultati, ma non sei riuscito a recuperare i testi
#   (id fuori range o simili), quindi context_str rimane vuota.
# - Metti un fallback più informativo.
#
# prompt = f"""..."""
# - Stringa multilinea (triple quotes) formattata con f-string.
# - Dentro inserisci:
#   - {context_str.strip()} -> contesto ripulito da spazi iniziali/finali
#   - {query} -> domanda utente
#
# Le regole 4 e 6 sono ridondanti (ripeti la stessa frase),
# ma in prompt engineering spesso la ripetizione aumenta aderenza.
#
# return prompt
# - Ritorni la stringa finale pronta per il LLM.


###############################################################################
# 10) CHIAMATA AL LLM: get_llm_response
###############################################################################
def get_llm_response(prompt: str, client_openai, llm_model_name: str = "gpt-3.5-turbo") -> str:
    if not client_openai:
        return "Errore (get_llm_response): Client OpenAI non configurato."
    if not prompt:
        return "Errore (get_llm_response): Prompt vuoto."

    try:
        response = client_openai.chat.completions.create(
            model=llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()

    except openai.AuthenticationError as e:
        print(f"ERRORE di Autenticazione OpenAI: {e}")
        return "Errore di Autenticazione OpenAI: Controlla la tua API Key e i permessi."
    except openai.RateLimitError as e:
        print(f"ERRORE OpenAI Rate Limit: {e}")
        return "Errore OpenAI: Raggiunto il limite di richieste (Rate Limit)."
    except openai.APIConnectionError as e:
        print(f"ERRORE di Connessione API OpenAI: {e}")
        return "Errore di Connessione API OpenAI."
    except openai.BadRequestError as e:
        print(f"ERRORE OpenAI Bad Request: {e}")
        return f"Errore OpenAI Bad Request: {e}"
    except Exception as e:
        error_message = f"Errore generico OpenAI ({llm_model_name}): {type(e).__name__} - {e}"
        print(error_message)
        return error_message

# --- SPIEGAZIONE MINUZIOSA ---
# get_llm_response:
# - Incapsula la chiamata al modello OpenAI e gestisce errori comuni.
#
# if not client_openai:
# - Se non hai client, ritorni subito un errore “stringa”.
# - Nota di stile: qui ritorni stringhe “errore”, non lanci eccezioni.
#   È una scelta: più semplice per demo, meno “pulita” per libreria.
#
# if not prompt:
# - Prompt vuoto: ritorni errore.
#
# response = client_openai.chat.completions.create(...)
# - Chiami endpoint chat completions.
# - Parametri:
#   - model: nome modello LLM
#   - messages: lista di messaggi in formato chat
#     qui: un solo messaggio user con tutto il prompt.
#   - temperature=0.0:
#     - risposta più deterministica e “fattuale”.
#     - riduce allucinazioni (non le elimina).
#   - max_tokens=350:
#     - limite lunghezza output (non input).
#     - utile per costi e per evitare risposte troppo lunghe.
#
# return response.choices[0].message.content.strip()
# - L’SDK ritorna una struttura con choices (lista).
# - Prendi la prima risposta.
# - strip() rimuove spazi/newline ai bordi.
#
# except openai.AuthenticationError ...
# except openai.RateLimitError ...
# - Gestione errori “comuni” con messaggi leggibili.
#
# except Exception as e:
# - fallback: cattura tutto.
# - In debug stampi anche il tipo di eccezione e messaggio.
#
# Nota compatibilità:
# - I nomi delle eccezioni e l’API possono cambiare tra versioni SDK.
# - La struttura generale però è questa.


###############################################################################
# 11) PIPELINE COMPLETA: run_rag_pipeline
###############################################################################
def run_rag_pipeline(query: str, index_data: list, model_emb, corpus_docs: list, client_oai,
                     llm_name: str = "gpt-3.5-turbo", num_top_k: int = 3) -> str:
    print(f"\n{'-'*20} AVVIO PIPELINE RAG {'-'*20}")
    print(f"Query Utente: '{query}'")

    if not index_data or not model_emb or not client_oai or not corpus_docs:
        error_msg = "Errore critico: componenti mancanti (indice/modello/client/corpus)."
        print(error_msg)
        return error_msg

    print(f"\n1. RECUPERO (Top-{num_top_k})...")
    search_results = search_semantic(query, index_data, model_emb, top_k=num_top_k)
    print(f"Recuperati {len(search_results)} documenti." if search_results else "Nessun documento recuperato.")

    print("\n2. AUMENTO (Prompt RAG)...")
    rag_prompt = build_rag_prompt(query, search_results, corpus_docs)

    print(f"\n3. GENERAZIONE (LLM: {llm_name})...")
    final_answer = get_llm_response(rag_prompt, client_oai, llm_name)

    print(f"\n{'-'*20} FINE PIPELINE RAG {'-'*20}")
    return final_answer

# --- SPIEGAZIONE MINUZIOSA ---
# run_rag_pipeline:
# - Orchestratore: collega Retrieval -> Prompt -> LLM.
# - Parametri:
#   - query: domanda utente
#   - index_data: indice embeddings
#   - model_emb: modello embedding
#   - corpus_docs: lista testi originali
#   - client_oai: client OpenAI
#   - llm_name: modello LLM
#   - num_top_k: quanti doc recuperare
#
# print(f"\n{'-'*20} ...")
# - '-'*20 crea "--------------------"
# - utile per separare le fasi a log.
#
# if not index_data or not model_emb or not client_oai or not corpus_docs:
# - Fail fast: se manca un componente, ritorni errore.
#
# 1) search_semantic(...)
# - Recupero documenti top-k
#
# 2) build_rag_prompt(...)
# - Costruisci prompt con contesto e regole.
#
# 3) get_llm_response(...)
# - Chiedi la risposta al LLM.
#
# return final_answer
# - Output finale della pipeline.


###############################################################################
# 12) TEST RAPIDO (OPZIONALE)
###############################################################################
if loaded_index and embedding_model and openai_client and documents_corpus:
    user_query_1 = "Cosa dicono i documenti sulla lealtà dei cani?"
    answer_1 = run_rag_pipeline(user_query_1, loaded_index, embedding_model, documents_corpus, openai_client, num_top_k=1)
    print(f"\n>>> RISPOSTA:\n{answer_1}")
else:
    print("\nTest pipeline saltato: setup incompleto.")

# --- SPIEGAZIONE MINUZIOSA ---
# if loaded_index and embedding_model and openai_client and documents_corpus:
# - Esegui il test SOLO se tutto è presente.
#
# user_query_1 = ...
# - Query di prova coerente col corpus (parla di cani e lealtà).
#
# answer_1 = run_rag_pipeline(..., num_top_k=1)
# - Recuperi solo 1 documento top-1.
# - Utile per test veloce: meno contesto, meno costo token LLM.
#
# print(f"\n>>> RISPOSTA:\n{answer_1}")
# - Stampa la risposta generata.
#
# else:
# - Se setup incompleto, eviti chiamate che fallirebbero.
###############################################################################
