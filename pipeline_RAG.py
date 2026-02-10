!pip install sentence-transformers -q
!pip install openai -q
!pip install scikit-learn -q

import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from google.colab import userdata # Per i segreti di Colab
import torch # Per verificare la GPU

print("Librerie importate.")

# Caricamento dell'indice vettoriale semplice
index_filepath = "my_simple_corpus_index.pkl"
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
    print(f"ERRORE: File indice '{index_filepath}' non trovato. Assicurati di aver eseguito L43 o carica il file.")

# Caricamento del Modello di Embedding
# DEVE ESSERE LO STESSO MODELLO USATO PER CREARE L'INDICE
# Modifica 'model_name_used_for_index' se hai usato un modello diverso.
model_name_used_for_index = 'paraphrase-multilingual-mpnet-base-v2'

embedding_model = None # Rinominato da 'model' a 'embedding_model' per chiarezza

try:
    # Carichiamo sempre il modello SentenceTransformer per chiarezza
    # ed evitare problemi di tipo con il riutilizzo di 'corpus_model' da lezioni precedenti.
    print(f"\nCaricamento del modello embedding '{model_name_used_for_index}'...")
    embedding_model = SentenceTransformer(model_name_used_for_index)
    # Usiamo la variabile model_name_used_for_index per confermare il nome,
    # poiché l'oggetto SentenceTransformer non ha un attributo .config.name_or_path diretto.
    print(f"Modello embedding '{model_name_used_for_index}' caricato con successo.")

    if embedding_model and torch.cuda.is_available():
        print(f"Il modello embedding sta utilizzando la GPU: {torch.cuda.get_device_name(0)}")
    elif embedding_model:
        print("Il modello embedding sta utilizzando la CPU.")

except Exception as e:
    print(f"ERRORE durante il caricamento del modello embedding '{model_name_used_for_index}': {e}")
    embedding_model = None


# Configurazione del Client OpenAI
openai_client = None
try:
    # Assicurati che la tua API Key sia stata aggiunta come segreto di Colab
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
    openai_client = openai.OpenAI()
    # Test rapido opzionale per verificare la connessione/chiave
    openai_client.models.list()
    print("\nClient OpenAI configurato correttamente usando i Segreti di Colab.")
except userdata.SecretNotFoundError:
    print("\nATTENZIONE: Segreto 'OPENAI_API_KEY' non trovato in Colab Secrets.")
    print("Per favore, aggiungi la tua API Key OpenAI ai Segreti di Colab (pannello a sinistra, icona chiave).")
    print("La parte di generazione LLM della pipeline RAG non funzionerà senza API Key.")
except openai.AuthenticationError:
    print("\nERRORE di Autenticazione OpenAI: La API Key fornita non è valida.")
    print("Verifica la API Key nei Segreti di Colab.")
except Exception as e:
    print(f"\nERRORE nella configurazione del client OpenAI: {e}")
    print("Verifica che la variabile d'ambiente OPENAI_API_KEY sia impostata correttamente se non usi i Segreti.")


# Recupero del Corpus Originale (necessario per il contesto)
# Questo è il corpus di testi che abbiamo indicizzato nella precedente lezione.
# Per semplicità didattica, lo ridefiniamo qui. Deve corrispondere alla lezione precedente.
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


# Controllo finale del setup
print("\n--- VERIFICA SETUP ---")
if not loaded_index: print("RISULTATO: Indice NON caricato.")
else: print("RISULTATO: Indice CARICATO.")
if not embedding_model: print("RISULTATO: Modello Embedding NON caricato.")
else: print("RISULTATO: Modello Embedding CARICATO.")
if not openai_client: print("RISULTATO: Client OpenAI NON configurato.")
else: print("RISULTATO: Client OpenAI CONFIGURATO.")
if not documents_corpus: print("RISULTATO: Corpus documenti NON disponibile.") # Dovrebbe essere sempre disponibile qui
else: print("RISULTATO: Corpus documenti DISPONIBILE.")

if not loaded_index or not embedding_model or not openai_client or not documents_corpus:
     print("\n--- ATTENZIONE: Setup incompleto. La pipeline RAG potrebbe non funzionare correttamente. Controlla i messaggi sopra. ---")
else:
     print("\n--- Setup completato con successo! Tutti i componenti per la pipeline RAG sono pronti. ---")

def search_semantic(query: str, index: list, emb_model, top_k: int = 3) -> list:
    """
    Esegue una ricerca semantica brute-force sull'indice fornito.

    Args:
        query: La stringa di query dell'utente.
        index: La lista indice [(embedding_vector, doc_reference), ...].
        emb_model: Il modello SentenceTransformer caricato.
        top_k: Il numero di risultati più simili da restituire.

    Returns:
        Una lista dei top_k risultati come tuple (score, doc_reference),
        o una lista vuota in caso di errore o nessun risultato.
    """
    if not index or not emb_model:
        print("Errore in search_semantic: Indice o modello di embedding non forniti.")
        return []
    if not query:
        print("Errore in search_semantic: Query vuota.")
        return []

    # Genera embedding per la query
    try:
        query_embedding = emb_model.encode([query])
    except Exception as e:
        print(f"Errore durante la generazione dell'embedding per la query '{query}': {e}")
        return []

    # Calcola similarità con tutti i documenti nell'indice
    all_similarities = []
    for doc_idx, (doc_embedding_vector, doc_reference) in enumerate(index):
        try:
            # Assicura che doc_embedding_vector sia un array numpy valido
            if not isinstance(doc_embedding_vector, np.ndarray):
                print(f"Attenzione (search_semantic): Trovato embedding non-ndarray nell'indice per rif. ID {doc_reference.get('id', 'N/A')} (elemento indice {doc_idx}). Salto.")
                continue

            # Riformatta doc_embedding_vector per cosine_similarity
            doc_embedding_reshaped = doc_embedding_vector.reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding_reshaped)[0][0]
            all_similarities.append( (similarity, doc_reference) )
        except Exception as e:
            print(f"Errore nel calcolo similarità per rif. ID {doc_reference.get('id', 'N/A')} (elemento indice {doc_idx}): {e}")

    # Ordina i risultati per similarità (decrescente)
    all_similarities.sort(key=lambda item: item[0], reverse=True)

    # Restituisci i top_k risultati
    return all_similarities[:top_k]

# Test rapido della funzione search_semantic
print("\n--- Test Funzione search_semantic ---")
if loaded_index and embedding_model:
    test_query_retrieval = "Parlami degli animali fedeli"
    test_results_retrieval = search_semantic(test_query_retrieval, loaded_index, embedding_model, top_k=2)

    if test_results_retrieval:
        print(f"Risultati del test per '{test_query_retrieval}':")
        for score, ref in test_results_retrieval:
            print(f"  - Score: {score:.4f}, ID: {ref.get('id', 'N/A')}, Preview: {ref.get('preview', 'N/A')}")
    else:
        print(f"  Nessun risultato trovato per '{test_query_retrieval}' o errore durante la ricerca.")
else:
    print("Impossibile testare search_semantic: indice o modello embedding mancanti (controlla Cella 1).")

def build_rag_prompt(query: str, search_results: list, original_documents_list: list) -> str:
    """
    Costruisce il prompt aumentato per l'LLM, includendo il contesto recuperato.

    Args:
        query: La query originale dell'utente.
        search_results: Lista di tuple (score, doc_reference) dalla ricerca semantica.
                        Ogni doc_reference dovrebbe avere un campo 'id'.
        original_documents_list: La lista originale dei documenti (usata per recuperare il testo completo via ID).

    Returns:
        La stringa del prompt RAG completo.
    """
    context_str = ""
    if not search_results:
        context_str = "Nessun contesto rilevante trovato nei documenti forniti.\n"
    else:
        # Costruisci la stringa del contesto dai documenti recuperati
        for i, (score, ref) in enumerate(search_results):
            doc_id = ref.get('id', -1) # Ottieni l'ID dal riferimento

            if 0 <= doc_id < len(original_documents_list):
                doc_text = original_documents_list[doc_id] # Recupera il testo completo dall'ID
                context_str += f"--- Contesto Documento {i+1} (ID: {doc_id}, Similarità Score: {score:.4f}) ---\n"
                context_str += doc_text + "\n"
                context_str += "--------------------------------------------------------------------------\n\n"
            else:
                print(f"Attenzione (build_rag_prompt): ID documento {doc_id} non valido o fuori range per riferimento: {ref}. Salto questo documento.")

    if not context_str.strip() and search_results: # Se i search_results c'erano ma nessun ID valido è stato trovato
        context_str = "Contesto rilevante trovato ma impossibile recuperare il testo dei documenti (problema con ID).\n"


    # Definisci il template del prompt RAG
    # Istruzioni chiare sono fondamentali per il grounding
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

# Test rapido della funzione build_rag_prompt---
print("\n--- Test Funzione build_rag_prompt ---")
# Usa i risultati del test di search_semantic, se disponibili
if 'test_results_retrieval' in locals() and test_results_retrieval and 'documents_corpus' in locals():
     test_prompt_rag = build_rag_prompt(test_query_retrieval, test_results_retrieval, documents_corpus)
     print(f"Test del prompt RAG per la query '{test_query_retrieval}':")
     # Stampa solo l'inizio e la fine per non mettere troppo testo a schermo
     print("INIZIO PROMPT RAG:\n" + "="*20 + "\n" + test_prompt_rag[:600] + "\n...\n" + "="*20 + "\nFINE PROMPT RAG (ultimi 150 caratteri):\n" + "="*20 + "\n" + test_prompt_rag[-150:])
elif not 'test_results_retrieval' in locals() or not test_results_retrieval:
     print("Impossibile testare build_rag_prompt: 'test_results_retrieval' non disponibile (controlla test Cella 2).")
else: # documents_corpus mancante
     print("Impossibile testare build_rag_prompt: 'documents_corpus' non disponibile (controlla Cella 1).")

def get_llm_response(prompt: str, client_openai, llm_model_name: str = "gpt-3.5-turbo") -> str:
    """
    Invia il prompt all'API di OpenAI e restituisce la risposta generata.

    Args:
        prompt: Il prompt completo (potenzialmente aumentato) da inviare.
        client_openai: Il client OpenAI configurato.
        llm_model_name: Il nome del modello LLM da usare (default: gpt-3.5-turbo).

    Returns:
        La risposta testuale generata dall'LLM o un messaggio di errore specifico.
    """
    if not client_openai:
        return "Errore (get_llm_response): Client OpenAI non configurato."
    if not prompt:
        return "Errore (get_llm_response): Prompt vuoto."

    try:
        response = client_openai.chat.completions.create(
            model=llm_model_name,
            messages=[
                # Potremmo aggiungere un messaggio di sistema (system) per istruzioni globali
                # sull'identità o il comportamento dell'assistente, ma per RAG, le istruzioni
                # dettagliate sono spesso meglio incluse direttamente nel prompt utente (come abbiamo fatto).
                {"role": "user", "content": prompt} # Il nostro prompt RAG completo va qui
            ],
            temperature=0.0, # Bassa temperatura per risposte deterministiche e basate sul contesto.
                             # Valori più alti (esempio 0.7) per più creatività, ma meno adatti a RAG basati fu fatti.
            max_tokens=350   # Limita la lunghezza massima della risposta.
        )
        # Accediamo al contenuto del messaggio di risposta dell'assistente
        return response.choices[0].message.content.strip()

    except openai.AuthenticationError as e:
         print(f"ERRORE di Autenticazione OpenAI: {e}")
         return "Errore di Autenticazione OpenAI: Controlla la tua API Key e i permessi."
    except openai.RateLimitError as e:
         print(f"ERRORE OpenAI Rate Limit: {e}")
         return "Errore OpenAI: Raggiunto il limite di richieste (Rate Limit). Riprova più tardi o controlla il tuo piano."
    except openai.APIConnectionError as e:
        print(f"ERRORE di Connessione API OpenAI: {e}")
        return "Errore di Connessione API OpenAI: Impossibile connettersi ai server OpenAI. Controlla la tua rete."
    except openai.BadRequestError as e: # Ad esempio, se il modello non esiste o il prompt è troppo lungo
        print(f"ERRORE OpenAI Bad Request: {e}")
        return f"Errore OpenAI Bad Request: {e}. Potrebbe essere un modello non valido o un prompt troppo lungo."
    except Exception as e:
        # Gestione più generica per altri possibili errori API
        error_message = f"Errore generico durante la chiamata all'LLM OpenAI ({llm_model_name}): {type(e).__name__} - {e}"
        print(error_message) # Logghiamo l'errore completo per debug
        return error_message

# Test rapido della funzione get_llm_response
print("\n--- Test Funzione get_llm_response (EFFETTUA CHIAMATA API!) ---")
if openai_client:
    # Usiamo un prompt molto semplice e breve per il test per non sprecare troppi token
    # e per essere sicuri che non sia il prompt RAG a causare problemi qui.
    simple_test_llm_prompt = "Ciao mondo! Come stai oggi?"
    print(f"Invio prompt di test semplice all'LLM: '{simple_test_llm_prompt}'")
    test_llm_response = get_llm_response(simple_test_llm_prompt, openai_client)
    print(f"Risposta LLM al test semplice: {test_llm_response}")
elif not openai_client:
    print("Impossibile testare get_llm_response: Client OpenAI non configurato (controlla Cella 1).")

def run_rag_pipeline(query: str,
                     index_data: list,
                     model_emb,
                     corpus_docs: list,
                     client_oai,
                     llm_name: str = "gpt-3.5-turbo",
                     num_top_k: int = 3) -> str:
    """
    Esegue l'intera pipeline RAG: Ricerca -> Costruzione Prompt -> Generazione LLM.

    Restituisce la risposta finale dell'LLM o un messaggio di errore.
    """
    print(f"\n{'-'*20} AVVIO PIPELINE RAG {'-'*20}")
    print(f"Query Utente: '{query}'")

    # Verifica prerequisiti essenziali per la pipeline
    if not index_data or not model_emb or not client_oai or not corpus_docs:
        error_msg = "Errore critico (run_rag_pipeline): Uno o più componenti necessari (indice, modello embedding, client OpenAI, corpus documenti) non sono disponibili."
        print(error_msg)
        return error_msg

    # RETRIEVAL: recupera i documenti più rilevanti
    print(f"\n1. Fase di RECUPERO (Top-{num_top_k} documenti)...")
    search_results = search_semantic(query, index_data, model_emb, top_k=num_top_k)

    if not search_results:
        print("   Nessun documento rilevante trovato nella knowledge base per questa query.")
        # Anche se non ci sono risultati, costruiamo comunque un prompt che lo indichi,
        # e l'LLM dovrebbe rispondere "informazioni non trovate" secondo le istruzioni.
    else:
        print(f"Recuperati {len(search_results)} documenti rilevanti.")

    # AUGMENTATION: costruisci il prompt aumentato
    print("\n2. Fase di AUMENTO (Costruzione Prompt RAG)...")
    rag_prompt = build_rag_prompt(query, search_results, corpus_docs)

    # GENERATION: ottieni la risposta dall'LLM
    print(f"\n3. Fase di GENERAZIONE (Chiamata a LLM: {llm_name})...")
    final_answer = get_llm_response(rag_prompt, client_oai, llm_name)

    print(f"\n{'-'*20} FINE PIPELINE RAG {'-'*20}")
    return final_answer

print("\n\n" + "="*70)
print("--- INIZIO TEST DELLA PIPELINE RAG COMPLETA ---")
print("="*70)

# Assicurati che tutti i componenti principali siano validi prima di eseguire i test
if loaded_index and embedding_model and openai_client and documents_corpus:
    print("\nTutti i componenti per il test della pipeline RAG sembrano essere pronti.\n")

    # Query 1 dovrebbe trovare informazioni pertinenti nei documenti (specifica)
    user_query_1 = "Cosa dicono i documenti sulla lealtà dei cani?"
    print(f"\n--- TEST 1: Query con info attese nel corpus (specifica sui cani) ---")
    answer_1 = run_rag_pipeline(query=user_query_1,
                                index_data=loaded_index,
                                model_emb=embedding_model,
                                corpus_docs=documents_corpus,
                                client_oai=openai_client,
                                num_top_k=1) # Recuperiamo solo il top 1 documento per questo test
    print(f"\n>>> RISPOSTA FINALE ALLA QUERY 1 ('{user_query_1}'):\n{answer_1}")

    print("\n" + "="*60 + "\n")

    # Query 2 su informazioni probabilmente non presenti nel corpus
    user_query_2 = "Qual è la capitale della Francia?"
    print(f"\n--- TEST 2: Query con info NON attese nel corpus ---")
    answer_2 = run_rag_pipeline(user_query_2, loaded_index, embedding_model, documents_corpus, openai_client, num_top_k=3)
    print(f"\n>>> RISPOSTA FINALE ALLA QUERY 2 ('{user_query_2}'):\n{answer_2}")

    print("\n" + "="*60 + "\n")

    # Query 3 su informazioni presenti, ma formulate in modo diverso (test per la ricerca semantica)
    user_query_3 = "Come si prepara un dolce che contiene mele?"
    print(f"\n--- TEST 3: Query semanticamente simile a info nel corpus ---")
    answer_3 = run_rag_pipeline(user_query_3, loaded_index, embedding_model, documents_corpus, openai_client, num_top_k=1)
    print(f"\n>>> RISPOSTA FINALE ALLA QUERY 3 ('{user_query_3}'):\n{answer_3}")

    print("\n" + "="*60 + "\n")

    # Query 4 su un argomento specifico presente in un solo documento
    user_query_4 = "Parlami del ciclo vitale degli insetti alati con metamorfosi completa."
    print(f"\n--- TEST 4: Query specifica su un documento ---")
    answer_4 = run_rag_pipeline(user_query_4, loaded_index, embedding_model, documents_corpus, openai_client, num_top_k=1)
    print(f"\n>>> RISPOSTA FINALE ALLA QUERY 4 ('{user_query_4}'):\n{answer_4}")

else:
    print("\nImpossibile eseguire i test completi della pipeline RAG: setup incompleto (controlla output Cella 1).")

print("\n\n" + "="*70)
print("--- FINE TEST DELLA PIPELINE RAG COMPLETA ---")
print("="*70)
