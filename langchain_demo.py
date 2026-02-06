from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Errore: variabile OPENAI_API_KEY non trovata.")
print("Chiave API trovata")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un assistente creativo e conciso."),
    ("human", "{question}")
])

llm_openai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)
chain_openai = prompt | llm_openai | StrOutputParser()

input_data = {"question": "Suggerisci 5 nomi creativi per un'azienda green. Uno per riga."}

print("\nEsecuzione della catena con ChatOpenAI...")
result_openai = chain_openai.invoke(input_data)
print("\nRisultato:")
print(result_openai)
