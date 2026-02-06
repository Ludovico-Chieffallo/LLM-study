"""
TUTORIAL (super commentato) — Memoria in LangChain con RunnableWithMessageHistory

Obiettivo:
1) Costruire una chat con LangChain che “ricorda” i messaggi precedenti.
2) Gestire più conversazioni (sessioni) con uno store: una memoria diversa per ogni session_id.
3) Capire bene CHI fa cosa:
   - ChatPromptTemplate: definisce la struttura dei messaggi (system + history + human)
   - RunnableWithMessageHistory: inserisce automaticamente la history nel prompt e salva i nuovi messaggi
   - InMemoryChatMessageHistory: è il "contenitore" in RAM che conserva i messaggi

NOTA IMPORTANTE:
- La memoria che usiamo qui è in RAM: se chiudi il programma, la memoria sparisce.
- In produzione useresti Redis, DB, file, ecc. Qui ci concentriamo sul concetto.
"""

import os
from dotenv import load_dotenv

# --- LangChain imports (chat model, prompt, memoria, wrapper) ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- OpenAI client (NON indispensabile per LangChain, ma lo lasciamo per mostrare che la key c'è) ---
from openai import OpenAI


# =========================================================
# 1) CARICAMENTO VARIABILI D'AMBIENTE (.env)
# =========================================================

print("Caricamento variabili d'ambiente...")

# load_dotenv() legge un file .env nella cartella corrente (o percorso standard)
# e carica le variabili definite lì nel sistema (es. OPENAI_API_KEY=...)
load_dotenv()

# recupero la key dalle variabili d'ambiente
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    # Se manca la key, non ha senso continuare, perché non potrai fare chiamate al modello
    raise ValueError("Errore: variabile OPENAI_API_KEY non trovata.")
else:
    print("Chiave API trovata")

    # Questa parte NON è strettamente necessaria per la memoria in LangChain.
    # Serve solo come "test" per vedere che la key esiste e che puoi creare un client OpenAI.
    try:
        client = OpenAI(api_key=api_key)
        print("Client OpenAI creato correttamente (test key OK).")
    except Exception as e:
        print("\nSi è verificato un errore nella creazione del client OpenAI:")
        print(repr(e))


# =========================================================
# 2) INIZIALIZZAZIONE LLM (MODELLO CHAT)
# =========================================================

print("\nInizializzazione della catena con memoria...")

llm = None

if os.getenv("OPENAI_API_KEY"):
    try:
        # ChatOpenAI è il wrapper LangChain per un modello chat di OpenAI.
        # model: quale modello usare
        # temperature: creatività (0 = più deterministico, 1 = più creativo)
        # max_tokens: limite di token in output (risposta)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)
        print("LLM ChatOpenAI inizializzato correttamente.")
    except Exception as e:
        print("\nErrore durante l'inizializzazione di ChatOpenAI:")
        print(repr(e))
else:
    print("Chiave API non trovata. Assicurati di avere la variabile OPENAI_API_KEY nel tuo .env.")


# =========================================================
# 3) PROMPT CON MEMORIA (HISTORY)
# =========================================================

print("Definizione del prompt con memoria...")

"""
Qui definisci la struttura dei messaggi passati al modello, in ordine.

- ("system", "..."): istruzioni permanenti
- MessagesPlaceholder(variable_name="history"): QUI entra la memoria (messaggi passati)
- ("human", "{input}"): l'ultimo messaggio dell'utente (variabile input)

⚠️ Punto chiave:
Il prompt dichiara che si aspetta due variabili:
- history
- input

Quindi se qualcuno invoca il prompt direttamente, deve passare entrambe.
Ma noi NON passeremo "history" a mano: lo farà RunnableWithMessageHistory.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un assistente creativo e conciso."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

print("ChatPromptTemplate con memoria definito correttamente.")


# =========================================================
# 4) STORE DELLA MEMORIA (UNA MEMORIA PER OGNI SESSIONE)
# =========================================================

print("\nConfigurazione store della memoria...")

"""
Qui creiamo uno "store" Python: un dizionario.
Chiave: session_id (stringa)
Valore: un oggetto InMemoryChatMessageHistory

InMemoryChatMessageHistory conserva i messaggi in una lista interna.
"""

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    LangChain chiamerà questa funzione ogni volta che serve la memoria per una sessione.

    Cosa deve fare:
    - Se non esiste una memoria per quel session_id, creala
    - Restituisci SEMPRE l'oggetto BaseChatMessageHistory associato a quella sessione
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

print("Store della memoria configurato correttamente.")


# =========================================================
# 5) COSTRUZIONE DELLA CHAIN E AGGIUNTA DEL WRAPPER DI MEMORIA
# =========================================================

print("\nComposizione della catena con memoria...")

conv_chain_with_history = None

if llm and prompt:
    # base_chain: "prompt | llm" significa:
    # - prendi il prompt (ChatPromptTemplate) -> genera i messaggi
    # - passali al modello (llm) -> ottieni la risposta AI
    base_chain = prompt | llm
    print("Base chain creata correttamente.")

    """
    Ora arriva la parte più importante per la MEMORIA:

    RunnableWithMessageHistory fa 2 cose fondamentali:
    A) Prima della chiamata:
       - recupera la memoria associata al session_id (tramite get_session_history)
       - inserisce la history nel prompt, nella variabile "history"
    B) Dopo la chiamata:
       - salva nello storico sia il messaggio dell’utente che la risposta dell’AI

    Parametri chiave:
    - get_session_history: come recuperare la memoria
    - input_messages_key: qual è la chiave del dizionario input che contiene il testo dell’utente
    - history_messages_key: qual è il nome della variabile nel prompt che ospita la history

    Nel tuo caso:
    - tu invocavi: {"input": "..."}  -> quindi input_messages_key="input"
    - nel prompt hai: MessagesPlaceholder(variable_name="history") -> history_messages_key="history"
    """

    conv_chain_with_history = RunnableWithMessageHistory(
        runnable=base_chain,
        get_session_history=get_session_history,
        history_messages_key="history",
        input_messages_key="input"
    )

    print("Catena con memoria composta correttamente.")
else:
    print("Impossibile comporre la catena con memoria. LLM o prompt non inizializzati correttamente.")
    print("Fine configurazione catena con memoria.")


# =========================================================
# 6) TEST: CONVERSAZIONE DI PROVA CON UNA SESSIONE
# =========================================================

print("--- Conversazione di Prova ---")

if conv_chain_with_history:
    # In una app vera questo ID arriverebbe da:
    # - cookie browser
    # - user_id
    # - token
    # - session server-side
    session_id_attuale = "chat_con_alessio_001"

    print(f"--- Inizio Conversazione (Session ID: {session_id_attuale}) ---")

    # ---------------------------
    # Primo input: introduzione
    # ---------------------------
    user_input_1 = "Ciao, sono uno studente di intelligenza artificiale e mi chiamo Alessio."
    print(f"Tu (Sessione: {session_id_attuale}): {user_input_1}")

    try:
        """
        invoke() richiede due cose:
        1) i dati di input della chain: {"input": ...}
        2) la config con session_id:
           config={"configurable": {"session_id": "..."}}

        Perché questa config?
        Perché RunnableWithMessageHistory deve sapere quale memoria usare.
        """
        ai_response_1 = conv_chain_with_history.invoke(
            {"input": user_input_1},
            config={"configurable": {"session_id": session_id_attuale}}
        )
        print(f"AI: {ai_response_1.content}")
    except Exception as e:
        print(f"ERRORE nell'invocare la chain (input 1): {e}")

    print("\n-----\n")

    # ---------------------------
    # Secondo input: verifica memoria (nome)
    # ---------------------------
    user_input_2 = "Come mi chiamo?"
    print(f"Tu (Sessione: {session_id_attuale}): {user_input_2}")

    try:
        ai_response_2 = conv_chain_with_history.invoke(
            {"input": user_input_2},
            config={"configurable": {"session_id": session_id_attuale}}
        )
        print(f"AI: {ai_response_2.content}")
    except Exception as e:
        print(f"ERRORE nell'invocare la chain (input 2): {e}")

    print("\n-----\n")

    # ---------------------------
    # Terzo input: verifica memoria (argomento studi)
    # ---------------------------
    user_input_3 = "Di cosa ti ho parlato riguardo ai miei studi?"
    print(f"Tu (Sessione: {session_id_attuale}): {user_input_3}")

    try:
        ai_response_3 = conv_chain_with_history.invoke(
            {"input": user_input_3},
            config={"configurable": {"session_id": session_id_attuale}}
        )
        print(f"AI: {ai_response_3.content}")
    except Exception as e:
        print(f"ERRORE nell'invocare la chain (input 3): {e}")

    print(f"\n--- Fine Conversazione (Session ID: {session_id_attuale}) ---")
else:
    print("Chain con memoria non inizializzata. Impossibile avviare la conversazione.")

print("--- Fine Conversazione di Prova ---\n")


# =========================================================
# 7) ISPEZIONE MEMORIA (COSA C'È DAVVERO NELLO STORE?)
# =========================================================

print("--- Ispezione Memoria (dallo store) ---")

"""
Qui guardiamo direttamente dentro lo store.
Se la chain ha funzionato, per session_id_attuale dovresti trovare:
- messaggio human 1
- risposta AI 1
- messaggio human 2
- risposta AI 2
- messaggio human 3
- risposta AI 3
Quindi tipicamente 6 messaggi totali.
"""

if 'session_id_attuale' in locals() and session_id_attuale in store:
    print(f"\n--- Contenuto della memoria per Session ID: {session_id_attuale} ---")

    specific_session_history = store[session_id_attuale]

    # .messages è la lista reale dei messaggi (HumanMessage/AIMessage/etc.)
    if not specific_session_history.messages:
        print("Il buffer di memoria per questa sessione è vuoto. (Hai eseguito la conversazione?)")
    else:
        print(f"Numero di messaggi nello storico per '{session_id_attuale}': {len(specific_session_history.messages)}")
        for m in specific_session_history.messages:
            role_emoji = "👤" if m.type == "human" else "🤖"
            print(f"{role_emoji} ({m.type}): {m.content}")
else:
    print(f"Cronologia non trovata per Session ID: {locals().get('session_id_attuale', 'NON DEFINITO')} nello store.")
    print(f"Lo store attualmente contiene cronologie per le seguenti session_ids: {list(store.keys())}")

print("--- Fine Ispezione Memoria ---\n")


"""
COSA HAI IMPARATO (riassunto concettuale):
1) Il prompt definisce dove va la memoria: MessagesPlaceholder("history")
2) Devi avere uno store per gestire più conversazioni (session_id -> history)
3) RunnableWithMessageHistory:
   - recupera history con get_session_history(session_id)
   - la passa al prompt dentro la variabile "history"
   - salva automaticamente nuovi messaggi (utente + AI)

SE VUOI ESTENDERE:
- sostituisci InMemoryChatMessageHistory con uno storage persistente (Redis/DB)
- aggiungi un "windowing": memorizza solo gli ultimi N messaggi per non crescere all'infinito
- aggiungi summarization: riassumi la storia quando diventa lunga
"""
