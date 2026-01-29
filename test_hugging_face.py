from transformers import pipeline
import logging

# Imposta il livello di log per la libreria 'transformers' a ERROR.
# Questo sopprime i messaggi di log di livello INFO e WARNING, rendendo l'output più pulito.
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    print("\nCaricamento pipeline Hugging Face...")
    print("Il primo avvio potrebbe richiedere il download del modello, attendi...")

    # Crea una pipeline di 'text-generation' utilizzando il modello pre-addestrato 'gpt2'.
    generator = pipeline('text-generation', model='gpt2')

    # Definisce il testo iniziale (prompt) da cui il modello genererà la continuazione.
    prompt = "Ciao dal mondo degli LLM, "
    print(f"\nGenero testo a partire da: '{prompt}'")

    # Utilizza la pipeline per generare testo a partire dal prompt.
    response = generator(prompt, max_length=50, num_return_sequences=1)

    print("\nRisposta del modello Hugging Face (GPT-2 locale): ")
    # Accediamo al primo elemento della lista (l'unica sequenza generata) e stampiamo il testo.
    print(response[0]['generated_text'])

# Gestisce l'errore di importazione che si verifica se la libreria 'transformers' o le sue dipendenze (come PyTorch o TensorFlow) non sono installate.
except ImportError:
    print("\nErrore: Libreria 'transformers' o 'torch'/'tensorflow' non trovata.")
    print("Assicurati di aver installato 'transformers' e una delle librerie di backend (torch o tensorflow).")
    print("Puoi installarle con: pip install transformers torch (per PyTorch) oppure pip install transformers tensorflow (per TensorFlow)")

# Gestisce eventuali altre eccezioni che potrebbero verificarsi durante l'utilizzo della libreria 'transformers'.
except Exception as e:
    print(f"\nSi è verificato un errore durante l'utilizzo di Hugging Face Transformers: {e}")
    print("Possibili cause: Modello non scaricabile (verifica la connessione internet e il nome del modello),")
    print("memoria insufficiente (prova a ridurre 'max_length' o usa un modello più piccolo),")
    print("problemi con le dipendenze (assicurati che tutte le librerie richieste siano installate correttamente).")