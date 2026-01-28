# Importiamo la classe AutoTokenizer dalla libreria transformers.
#
# transformers è la libreria di Hugging Face che contiene:
# - modelli pre-addestrati (BERT, GPT, ecc.)
# - tokenizer
# - pipeline NLP
#
# AutoTokenizer è una "factory class":
# in base al nome del modello che gli passiamo,
# capisce automaticamente quale tipo di tokenizer caricare
# (WordPiece, BPE, SentencePiece, ecc.).
from transformers import AutoTokenizer


# Qui carichiamo un tokenizer pre-addestrato associato al modello:
# "bert-base-uncased".
#
# - "bert" = tipo di modello (Bidirectional Encoder Representations from Transformers)
# - "base" = dimensione standard (12 layer, ~110M parametri)
# - "uncased" = non fa distinzione tra maiuscole e minuscole
#               → "Hello" e "hello" diventano la stessa cosa.
#
# Internamente succede:
# 1) HuggingFace controlla se il tokenizer è già nella cache locale.
# 2) Se non c'è, lo scarica dai server.
# 3) Carica:
#    - vocabolario (vocab.txt)
#    - regole di tokenizzazione
#    - configurazione
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Definiamo una stringa di testo in input.
# Questa è la frase che vogliamo tokenizzare.
#
# Nota: BERT-base-uncased è addestrato soprattutto su inglese,
# quindi l'italiano funzionerà ma non in modo perfetto.
testo = "Ciao, come stai?"


# Qui chiamiamo il metodo tokenize().
#
# tokenize():
# - prende la stringa
# - la normalizza (minuscole, rimozione accenti se previsto, ecc.)
# - la spezza in "subword tokens"
#
# I subword sono pezzi di parola:
# invece di avere un token per ogni parola,
# il vocabolario contiene parti frequenti di parole.
#
# Esempio tipico:
#   "playing" -> ["play", "##ing"]
#
# Il prefisso "##" indica che il token è
# una continuazione della parola precedente.
tokens = tokenizer.tokenize(testo)


# Qui usiamo encode().
#
# encode() fa PIÙ cose insieme:
# 1) tokenizza la frase (come tokenize())
# 2) aggiunge token speciali del modello, ad esempio:
#    - [CLS] → inizio sequenza
#    - [SEP] → fine sequenza
# 3) converte ogni token nel suo ID numerico
#    usando il vocabolario interno
#
# Il risultato è una lista di interi.
#
# Questi numeri sono ciò che il modello neurale
# riceve effettivamente in input.
input_ids = tokenizer.encode(testo)


# Stampiamo il testo originale, così possiamo confrontarlo.
print("testo originale:", testo)


# Stampiamo la lista di token (stringhe).
#
# Qui vedrai qualcosa tipo:
# ['ciao', ',', 'come', 'stai', '?']
# (dipende dal tokenizer).
print("Tokens (subwords):", tokens)


# Stampiamo la lista degli ID numerici.
#
# Ogni numero corrisponde a un token del vocabolario BERT.
# Gli ID includeranno anche quelli di [CLS] e [SEP].
print("Token IDs:", input_ids)
