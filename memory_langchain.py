import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
print("Caricamento variabili d'ambiente...")

load_dotenv()   
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

print("\nInizializzazione della catena con memoria...")
llm= None
if os.getenv("OPENAI_API_KEY"):
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)
        print("LLM ChatOpenAI inizializzato correttamente.")
    except Exception as e:
        print("\nSi è verificato un errore durante l'inizializzazione di ChatOpenAI:")
        print(repr(e))
else:
    print("Chiave API non trovata. Assicurati di avere la variabile OPENAI_API_KEY nel tuo .env.")

