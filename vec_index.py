###############################################################################
# OBIETTIVO GENERALE DELLO SCRIPT
###############################################################################
# Questo script fa (in ordine):
# 1) Definisce un piccolo "corpus" di documenti testuali (lista di frasi).
# 2) Carica un modello Sentence-Transformers (un encoder di frasi).
# 3) Converte ogni documento in un vettore numerico (embedding).
# 4) Costruisce un "indice" molto semplice: una lista di tuple (embedding, metadati_doc).
# 5) Salva l'indice su disco con pickle.
# 6) Ricarica l'indice da disco e stampa alcune informazioni per verifica.
#
# Contesto LLM/Embedding:
# - Qui NON stai "addestrando" un LLM. Stai usando un modello pre-addestrato per creare
#   embeddings, cioè rappresentazioni vettoriali utili per similarity search, retrieval,
#   clustering, ecc.
###############################################################################


###############################################################################
# 1) IMPORT DELLE LIBRERIE
###############################################################################
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pickle
import os

print("libreries imported successfully")

# --- SPIEGAZIONE RIGA PER RIGA ---
# import numpy as np
# - NumPy è la libreria standard per array e calcoli numerici in Python.
# - Qui serve POTENZIALMENTE perché gli embeddings spesso sono array NumPy.
# - Nota: nel tuo script, np non viene mai usato esplicitamente dopo l'import.
#   Quindi è "superfluo" ma non dannoso. Spesso si importa per abitudine.

# import torch
# - PyTorch è il framework di deep learning che many SentenceTransformers usa sotto al cofano.
# - Qui lo usi per controllare se esiste la GPU (torch.cuda.is_available()) e, in teoria,
#   per spostare il modello su CUDA.

# from sentence_transformers import SentenceTransformer
# - Importi la classe principale per caricare un modello encoder di frasi.
# - Serve per: corpus_model = SentenceTransformer(model_name_to_load)

# import pickle
# - Pickle è un modulo Python per serializzare (salvare) oggetti su disco e ricaricarli.
# - Qui lo usi per salvare la struttura "simple_index" in un file .pkl.

# import os
# - Serve per interagire con filesystem/sistema operativo.
# - Qui lo usi per: os.path.exists(index_filepath)


###############################################################################
# 2) DEFINIZIONE CORPUS DI ESEMPIO
###############################################################################
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

# --- SPIEGAZIONE MINUZIOSA ---
# documents = [ ... ]
# - Una lista Python di stringhe. Ogni stringa è un "documento".
# - In un progetto reale, potrebbe arrivare da file, DB, web scraping, ecc.

# ATTENZIONE IMPORTANTISSIMA (BUG LOGICO NEL CORPUS):
# Hai scritto:
#   "the human brain ... emotion."
#   "the history of the Roman Empire ..."
# senza virgola tra le due stringhe.
#
# In Python, due stringhe letterali adiacenti vengono CONCATENATE automaticamente:
#   "a" "b"  ->  "ab"
#
# Quindi quel punto della lista NON contiene due documenti: ne contiene UNO solo
# che è la concatenazione delle due frasi.
#
# Risultato: len(documents) sarà 9 e non 10 (a meno che tu non abbia 10 altrove).
# Perché uno degli elementi è stato “fuso” con quello dopo.
#
# print(f"Number of documents: {len(documents)}")
# - Serve a verificare rapidamente quanti documenti ci sono.

# for i, doc in enumerate(documents):
# - enumerate ti dà (indice, valore). Qui indice i e documento doc.

# print(f"Document {i}: {doc[:50]}...")
# - doc[:50] è slicing: prendi i primi 50 caratteri, utile per non stampare tutto.
# - ... è solo estetica: fa capire che stai mostrando un’anteprima.


###############################################################################
# 3) SCELTA DEL MODELLO DA CARICARE (Sentence-Transformers)
###############################################################################
# Load the pre-trained model
model_name_to_load = "paraphrase-multilingual-mpnet-base-v2"

# --- SPIEGAZIONE ---
# model_name_to_load è una stringa con il nome del modello su HuggingFace / SentenceTransformers.
# "paraphrase-multilingual-mpnet-base-v2" è un encoder multilingua.
# Perché sceglierlo?
# - Multilingua: funziona bene anche con testo non inglese (italiano incluso).
# - "paraphrase" indica che è addestrato a mettere frasi semanticamente simili vicine nello
#   spazio vettoriale: perfetto per similarity search.


###############################################################################
# 4) CACHING DEL MODELLO IN MEMORIA (uso di globals())
###############################################################################
corpus_model = None
current_model_name_in_memory = None

if "active_corpus_model" in globals() and "active_model_name" in globals():
    if active_model_name == model_name_to_load:
        corpus_model = active_corpus_model
        current_model_name_in_memory = active_model_name
    else:
        print(f"Il modello attivo in memoria ({active_model_name}) è diverso da quello richiesto ({model_name_to_load}). Caricamento del nuovo modello...")

# --- SPIEGAZIONE MINUZIOSA (QUI C'È UN CONCETTO DI "CACHE") ---
# corpus_model = None
# - Inizializzi la variabile che conterrà il modello vero e proprio.
# - None è un valore sentinella (significa: "non ancora disponibile").

# current_model_name_in_memory = None
# - Variabile per ricordare quale modello è effettivamente in uso.

# if "active_corpus_model" in globals() and "active_model_name" in globals():
# - globals() è un dizionario con tutte le variabili globali del file/ambiente.
# - Questo blocco serve tipicamente in notebook (Jupyter/Colab):
#   se esegui la cella più volte, vuoi evitare di ricaricare il modello ogni volta.
# - Quindi: se esistono già variabili globali "active_corpus_model" e "active_model_name",
#   provi a riutilizzarle.

# if active_model_name == model_name_to_load:
# - Confronti il nome del modello già in memoria con quello richiesto.
# - Se coincidono: riusi l'oggetto modello già caricato (risparmi tempo e RAM/VRAM).
#
# corpus_model = active_corpus_model
# - Prendi il modello dalla variabile globale e lo metti nella variabile locale.
# - Motivo: lavorare con una variabile locale è più pulito e riduce dipendenze globali.
#
# else: print(...)
# - Se in memoria c’è un modello diverso, avvisi che caricherai quello richiesto.

# NOTA DI CODING:
# - Questo approccio con globals() funziona, ma è un po’ “sporco” in progetti seri.
# - In un progetto reale useresti una classe, un singleton, o una cache controllata.


###############################################################################
# 5) CARICAMENTO DEL MODELLO SE NON È GIÀ IN MEMORIA
###############################################################################
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

# --- SPIEGAZIONE MINUZIOSA ---
# if corpus_model is None:
# - Significa: "non ho trovato/riusato un modello in cache", quindi devo caricarlo.

# print(...)
# - Messaggio di log: utile per capire cosa succede (soprattutto se il caricamento è lento).

# try:
# - Gestione errori: il caricamento può fallire per tanti motivi:
#   - modello non scaricabile / offline
#   - problemi di versione
#   - mancanza di dipendenze
#   - spazio disco insufficiente
#   - ecc.

# from sentence_transformers import SentenceTransformer
# - Re-import dentro il try. Non è necessario (l’hai già importato sopra),
#   ma serve a catturare eventuali problemi di import proprio qui.
# - In pratica: ridondante, ma a volte usato didatticamente.

# corpus_model = SentenceTransformer(model_name_to_load)
# - Qui succede la magia:
#   - se il modello non è in cache locale, lo scarica (o prova).
#   - costruisce il modello PyTorch interno.
# - Risultato: un oggetto pronto a fare encode().

# current_model_name_in_memory = model_name_to_load
# - Ora sai che il modello in uso ha quel nome.

# globals()['active_corpus_model'] = corpus_model
# globals()['active_model_name'] = current_model_name_in_memory
# - Salvi in globals la cache: così se riesegui, puoi riusarlo.
# - È una scelta tipica da notebook.

# ORDINE DEGLI except (nota molto importante):
# Nel tuo codice hai:
#   except Exception as e:
#   except ImportError as e:
# Ma ImportError è una sottoclasse di Exception.
# Quindi il blocco ImportError NON verrà MAI raggiunto:
# qualsiasi ImportError verrebbe catturato dal primo except Exception.
#
# Per essere corretti, l'ordine dovrebbe essere:
#   except ImportError as e:
#   except Exception as e:


###############################################################################
# 6) SPOSTAMENTO DEL MODELLO SU GPU (SE DISPONIBILE)
###############################################################################
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

# --- SPIEGAZIONE MINUZIOSA ---
# if corpus_model:
# - In Python, un oggetto non-None è "truthy".
# - Quindi questo blocco entra solo se il modello esiste davvero.

# if torch.cuda.is_available():
# - Controlla se PyTorch vede una GPU CUDA configurata correttamente.

# try:
# - Spostare su GPU può fallire per:
#   - driver/compatibilità
#   - memoria GPU insufficiente
#   - modello non spostabile in quel modo (raro, ma possibile)

# device = next(corpus_model.parameters()).device
# - corpus_model.parameters() restituisce un iteratore sui parametri (tensori) del modello.
# - next(...) prende il primo parametro.
# - .device ti dice dove si trova (cpu o cuda:0, ecc.)
# - Motivo della scelta: è un modo rapido per dedurre la device del modello.

# if "cuda" in str(device):
# - Converte il device in stringa e verifica se contiene "cuda".
# - È un controllo semplice (un po’ “grezzo”), ma funziona.
# - Alternativa più pulita: device.type == "cuda"

# corpus_model = corpus_model.to('cuda')
# - Sposta il modello su GPU.
# - Perché farlo? Per velocizzare l’encoding, specie con molti documenti.

# else: "GPU non disponibile, utilizzando CPU."
# - Non è un errore: su CPU funziona comunque, solo più lento.

# else finale (modello non caricato):
# - Se corpus_model è None o non valido, non puoi fare nulla.


###############################################################################
# 7) GENERAZIONE DEGLI EMBEDDINGS DEL CORPUS
###############################################################################
corpus_embeddings = None
if corpus_model:
    # modifica la riga per usare current_model_name_in_memory
    print (f"\nGenerazione embeddings per i{len(documents)} documenti usando il modello '{current_model_name_in_memory}'...")
    # show progress bar è utile per documenti più grandi.
    corpus_embeddings = corpus_model.encode(documents, show_progress_bar=True)

    print("Embeddings generati con successo.")
    print(f"Shape degli embeddings: {corpus_embeddings.shape}")

    if corpus_embeddings is not None:
        print(f"Dimensione di un singolo embedding: {corpus_embeddings[0].shape[0]}")
        print(f"Primi 3 valori del primo embedding: {corpus_embeddings[0][:3]}...")
else:
    print("Il modello non è stato caricato correttamente, non è possibile generare gli embeddings.")

# --- SPIEGAZIONE MINUZIOSA ---
# corpus_embeddings = None
# - Prepari una variabile che conterrà una matrice: un embedding per documento.

# if corpus_model:
# - Solo se il modello esiste.

# print(f"\nGenerazione embeddings per i{len(documents)} ...")
# - \n crea una riga vuota prima: migliora leggibilità log.
# - NOTA: hai scritto "per i{len(documents)}" senza spazio:
#   stamperà "per i9" invece di "per i 9". Solo estetica.

# corpus_embeddings = corpus_model.encode(documents, show_progress_bar=True)
# - encode prende una lista di stringhe e restituisce embeddings.
# - show_progress_bar=True:
#   - utile quando i documenti sono tanti (mostra barra di avanzamento).
#   - qui non è necessario, ma non dà fastidio.
#
# Importante: cosa torna encode()?
# - Di default, spesso torna un np.ndarray di shape (N, D)
#   dove:
#   - N = numero documenti
#   - D = dimensione embedding (es. 768)
#
# print(f"Shape degli embeddings: {corpus_embeddings.shape}")
# - .shape è una proprietà tipica di array NumPy: (N, D)

# if corpus_embeddings is not None:
# - Ridondante perché se encode non fallisce, non è None.
# - Ma è una "difesa" in più.

# corpus_embeddings[0].shape[0]
# - corpus_embeddings[0] è l'embedding del primo documento: un vettore lungo D.
# - .shape[0] è la lunghezza del vettore (dimensione embedding).

# corpus_embeddings[0][:3]
# - slicing: primi 3 valori del vettore (solo preview).


###############################################################################
# 8) CREAZIONE DI UN INDICE SEMPLICE (lista di tuple)
###############################################################################
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

# --- SPIEGAZIONE MINUZIOSA ---
# simple_index = []
# - Crei una lista vuota che conterrà una voce per documento.
# - "Indice" qui non è un vero index vettoriale (tipo FAISS).
# - È solo una struttura dati comoda per associare embedding + metadati.

# if corpus_embeddings is not None:
# - Se embeddings esistono.

# for i, embedding in enumerate(corpus_embeddings):
# - Iteri su ogni embedding.
# - i è l'indice del documento.
# - embedding è un vettore (array) di dimensione D.

# doc_reference = { ... }
# - Crei un dizionario di metadati:
#   "id": i -> id numerico
#   "text": documents[i] -> testo completo
#   "preview": anteprima max 80 char -> utile in UI/log

# "preview": documents[i] if len(documents[i]) <= 80 else documents[i][:80] + "..."
# - Operatore ternario in Python:
#   valore_se_vero if condizione else valore_se_falso
# - Serve per evitare preview più lunghe di 80 caratteri.

# simple_index.append((embedding, doc_reference))
# - Inserisci una tuple (embedding, reference).
# - Perché tuple?
#   - È immutabile (per convenzione: "questa coppia appartiene insieme").
#   - È veloce e semplice.
# - Perché non dict con chiavi?
#   - Si potrebbe fare, ma qui tuple è minimalista.

# if simple_index:
# - Lista non vuota => truthy.
# - Stampi un esempio della prima voce per ispezione.


###############################################################################
# 9) SALVATAGGIO DELL’INDICE SU DISCO CON PICKLE
###############################################################################
index_filepath = "simple_index.pkl"

if simple_index:
    try:
        with open(index_filepath, 'wb') as f_out:
            pickle.dump(simple_index, f_out, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Indice salvato con successo in '{index_filepath}'.")
    except Exception as e:
        print(f"Si è verificato un errore durante il salvataggio dell'indice: {repr(e)}")
else:
    print("L'indice non è stato creato correttamente, non è possibile salvarlo.")

# --- SPIEGAZIONE MINUZIOSA ---
# index_filepath = "simple_index.pkl"
# - Percorso del file di output.
# - ".pkl" è una convenzione per file pickle.

# if simple_index:
# - Salvi solo se l'indice non è vuoto.

# with open(index_filepath, 'wb') as f_out:
# - open(..., 'wb') = write binary (scrittura binaria).
# - Pickle produce bytes, quindi serve modalità binaria.
# - with garantisce la chiusura automatica del file anche se c'è errore.

# pickle.dump(simple_index, f_out, protocol=pickle.HIGHEST_PROTOCOL)
# - dump serializza l'oggetto e lo scrive nel file.
# - protocol=HIGHEST_PROTOCOL usa il protocollo più recente disponibile:
#   - file più efficiente/compatto
#   - ma meno compatibile con Python molto vecchi.

# NOTA DI SICUREZZA:
# - pickle NON è sicuro se carichi file da fonti non fidate.
# - Un pickle malevolo può eseguire codice. Qui va bene perché lo crei tu.


###############################################################################
# 10) CARICAMENTO DELL’INDICE DA DISCO E VERIFICA
###############################################################################
loaded_index = None
if os.path.exists(index_filepath):
    try:
        with open(index_filepath, 'rb') as f_in:
            loaded_index = pickle.load(f_in)
        print(f"Indice caricato con successo da '{index_filepath}'.")

        if loaded_index:
            print(f"Numero di voci nell'indice caricato: {len(loaded_index)}")
            print("Struttura di una voce dell'indice caricato:")
            first_loaded_embedding, first_loaded_reference = loaded_index[0]
            print(f"  - Tipo di embedding: {type(first_loaded_embedding)}")
            print(f"  - Dimensione dell'embedding: {first_loaded_embedding.shape}")
            print(f"  - Contenuto del riferimento: {first_loaded_reference}")
            print(f"  - Testo di anteprima: {first_loaded_reference['preview']}")
            print(f"  - Primi 3 valori dell'embedding: {first_loaded_embedding[:3]}...")
    except Exception as e:
        print(f"Si è verificato un errore durante il caricamento dell'indice: {repr(e)}")
else:
    print(f"Il file '{index_filepath}' non esiste, non è possibile caricare l'indice.")

# --- SPIEGAZIONE MINUZIOSA ---
# loaded_index = None
# - Variabile per mettere l'oggetto ricaricato.

# if os.path.exists(index_filepath):
# - Controllo preventivo: evita errore FileNotFoundError e ti permette un messaggio più chiaro.

# with open(index_filepath, 'rb') as f_in:
# - 'rb' = read binary, perché pickle lavora su bytes.

# loaded_index = pickle.load(f_in)
# - Deserializza l’oggetto dal file e lo ricostruisce in memoria.
# - Se hai salvato una lista di tuple, ottieni di nuovo una lista di tuple.

# if loaded_index:
# - Se non è vuoto, stampi un esempio.
# - Le stampe verificano che la struttura sia coerente con quella originale.

###############################################################################
# FINE SCRIPT
###############################################################################
