import os
from openai import OpenAI
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





def call_openai(prompt, max_tokens_ris=150):
    if not client:
        raise ValueError("Client OpenAI non inizializzato correttamente.")
    print(f"\n Invio prompt(max_tokens={max_tokens_ris})")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # <-- usa un modello esistente su OpenAI
            messages=[
                {"role": "system", "content": "Sei un assistente AI cattivo e antipatico."},
                {"role": "user", "content": prompt}
            ],
            max_tokens = max_tokens_ris,
            temperature=0.7
        )
        text_genered = response.choices[0].message.content.strip()
        print(f"\nRisposta dal modello OpenAI({response.model})")
        print(f"token prompt: {response.usage.prompt_tokens}, token risposta: {response.usage.completion_tokens}")
        print(f"token usati: {response.usage.total_tokens}")
        return text_genered        
    except openai.APIError as api_error:
        print("\nSi è verificato un errore API durante la chiamata all'API di OpenAI:")
        print(repr(api_error))
    except Exception as e:
        print("\nSi è verificato un errore durante la chiamata all'API di OpenAI:")
        print(repr(e))
   

print(call_openai("salutami"))
    
    