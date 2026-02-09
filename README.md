# 🧠 LangChain Memory Chain — Starter Guide & Checklist

Questo repository mostra come costruire una chat con memoria in LangChain usando:

- ChatOpenAI
- ChatPromptTemplate + MessagesPlaceholder
- RunnableWithMessageHistory
- InMemoryChatMessageHistory

L’obiettivo è fornire:

- una guida riutilizzabile
- una checklist operativa
- un punto di riferimento per progetti futuri
- debugging rapido
- struttura pronta per estensioni avanzate

---

## 🚀 Cosa imparerai

In questo progetto vedrai come:

- creare una chain LangChain con stato conversazionale
- gestire più sessioni
- collegare la memoria al prompt
- passare il session_id
- ispezionare lo storico
- evitare errori comuni

---

## 📦 Stack Tecnologico

- Python 3.10+
- LangChain
- langchain-core
- langchain-openai
- python-dotenv
- openai

---

## 🔐 Configurazione API Key

Crea un file .env nella root:

OPENAI_API_KEY=your_api_key_here

⚠️ Non committare mai .env.

Aggiungilo a .gitignore.

---

## 📥 Installazione

Installa le dipendenze:

pip install -r requirements.txt

### requirements.txt minimo

langchain
langchain-core
langchain-openai
python-dotenv
openai

---

# ✅ To-Do List — Costruire una Chain con Memoria

Usa questa checklist ogni volta che inizi un nuovo progetto.

---

## 1️⃣ Caricare le Variabili d’Ambiente

- load_dotenv()
- os.getenv("OPENAI_API_KEY")
- interrompere se la chiave non esiste

---

## 2️⃣ Inizializzare il Modello

- ChatOpenAI(...)
- configurare:
  - model
  - temperature
  - max_tokens

---

## 3️⃣ Creare un Prompt Compatibile con la Memoria

Il prompt deve contenere:

- ("system", "...")
- MessagesPlaceholder("history")
- ("human", "{input}")

Regola fondamentale:

Il nome history deve combaciare con:

history_messages_key="history"

Il nome input deve combaciare con:

input_messages_key="input"

---

## 4️⃣ Creare lo Store delle Sessioni

- dizionario store = {}
- funzione:

def get_session_history(session_id: str):

- se la sessione non esiste → crearla
- restituire sempre BaseChatMessageHistory

---

## 5️⃣ Costruire la Base Chain

base_chain = prompt | llm

---

## 6️⃣ Wrappare la Chain con la Memoria

Usare:

RunnableWithMessageHistory(
runnable=base_chain,
get_session_history=get_session_history,
input_messages_key="input",
history_messages_key="history"
)

---

## 7️⃣ Invocare la Chain con Session ID

Ogni chiamata deve includere:

chain.invoke(
{"input": user_text},
config={"configurable": {"session_id": session_id}}
)

Senza session_id la memoria non funziona.

---

## 8️⃣ Testare la Memoria

Esegui almeno 3 turni:

- Mi chiamo X
- Come mi chiamo?
- Di cosa ti ho parlato?

Il modello deve rispondere usando lo storico.

---

## 9️⃣ Ispezionare lo Store

Controlla:

- store[session_id].messages
- ruoli
- contenuti
- numero messaggi (2 per turno)

---

# 🛠 Troubleshooting

---

## Errore: missing variable history

Cause tipiche:

- placeholder non collegato
- chiave errata nel wrapper
- mismatch tra nomi

Verifica:

- MessagesPlaceholder("history")
- history_messages_key="history"

---

## Memoria Vuota

Controlla:

- stai passando session_id?
- la chain viene davvero eseguita?
- lo store è globale?

---

# 🚀 Estensioni Future

Quando vuoi portare questo setup a livello production:

- Redis / database
- memoria persistente
- sliding window memory
- summarization memory
- RAG + memoria
- tool calling
- session management avanzato

---

# 📌 Regole d’Oro

✔ Il prompt deve contenere la history  
✔ Il wrapper deve iniettarla  
✔ Ogni chiamata passa session_id  
✔ .env non va versionato  

---

# 📜 Licenza

Scegli quella che preferisci:

- MIT
- Apache-2.0

---

Happy hacking with LangChain
