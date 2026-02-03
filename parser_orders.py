import openai
import os
import json

try:
    os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI()
    print("OpenAI client initialized successfully.")
except KeyError:
    raise KeyError("OPENAI_API_KEY environment variable not set.")

input_text = """
Gentile Mario Rossi,
La contattiamo per il suo ordine #12345 effettuato il 15 Marzo 2024.
Vorremmo informarla che il suo ordine è stato spedito e dovrebbe arrivare entro
può contattarci per ulteriori informazioni a support@example.com.
Cordiali saluti
"""

prompt_json = f"""
dato il seguente testo: 
-----
{input_text}
-----
estrai il nome del destinatario, il numero dell'ordine, la data dell'ordine, lo stato della spedizione e l'indirizzo email di contatto.
Rispondi **solo** con oggetto JSON valido che contenga le seguenti chiavi: "nome_destinatario", "numero_ordine", "data_ordine", "stato_spedizione", "email_contatto".
Non Aggiungere niente altro al di fuori dell'oggetto JSON.
esempio di formato di risposta:
{{
  "nome_destinatario": "Nome Cognome",
  "numero_ordine": "numero",
  "data_ordine": "giorno mese anno",
  "stato_spedizione": "spedito/non spedito",
  "email_contatto": email@example.com"}}
"""
print("Prompt JSON:", prompt_json)

try:
    response= client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sei un assistente che estrae informazioni da testi in italiano.",
             "role": "user", "content": prompt_json}],
        temperature=0.1,
    )

    response_text = response.choices[0].message.content.strip()
    print("Response Text:", response_text)
except Exception as e:
    raise RuntimeError(f"Errore durante la chiamata all'API OpenAI: {e}")   
extracted_data = None
try:
    extracted_data = json.loads(response_text)
    print("Extracted Data:", extracted_data)
    name = extracted_data.get("nome_destinatario", "N/A")
    order_number = extracted_data.get("numero_ordine", "N/A")
    order_date = extracted_data.get("data_ordine", "N/A")
    shipping_status = extracted_data.get("stato_spedizione", "N/A")
    contact_email = extracted_data.get("email_contatto", "N/A")
    print(f"Nome Destinatario: {name}")
    print(f"Numero Ordine: {order_number}")
    print(f"Data Ordine: {order_date}")
    print(f"Stato Spedizione: {shipping_status}")
    print(f"Email Contatto: {contact_email}")
except json.JSONDecodeError as e:
    raise ValueError(f"Errore nel parsing della risposta JSON: {e}")
