
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # carica .env dalla root del progetto

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Errore: variabile OPENAI_API_KEY non trovata.")
    print("Hai creato il file .env nella root con OPENAI_API_KEY=<la_tua_chiave>?")
else:
    try:
        # Puoi anche fare: client = OpenAI(api_key=api_key)
        client = OpenAI()

        print("Invio richiesta all'API di OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # <-- usa un modello esistente su OpenAI
            messages=[
                {"role": "system", "content": "Sei un assistente AI simpatico."},
                {"role": "user", "content": "Scrivi un breve saluto dal mondo degli LLM."}
            ],
            max_tokens=50,
            temperature=0.7
        )

        print("\nRisposta dal modello OpenAI:")
        print(response.choices[0].message.content.strip())

    except Exception as e:
        print("\nSi è verificato un errore durante la chiamata all'API di OpenAI:")
        print(repr(e))

