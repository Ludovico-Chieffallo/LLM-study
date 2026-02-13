###############################################################################
# OBIETTIVO GENERALE DELLO SCRIPT
###############################################################################
# Questo script fa un fine-tuning (in realtà: “language modeling training”) di GPT-2
# sul dataset WikiText-2 (versione raw), usando Hugging Face Transformers + Datasets.
#
# Pipeline:
# 1) Carica dataset wikitext-2 e ne prende un subset piccolo (500 train, 100 eval).
# 2) Carica tokenizer e modello GPT-2.
# 3) Tokenizza i testi e crea le labels per causal language modeling (labels = input_ids).
# 4) Configura TrainingArguments.
# 5) Crea Trainer e avvia trainer.train().
# 6) Salva il modello fine-tuned e tokenizer.
# 7) Usa pipeline("text-generation") per generare testo con il modello salvato.
#
# Concetto chiave:
# - GPT2LMHeadModel è un modello per “Causal Language Modeling”:
#   impara a predire il token successivo dato il contesto precedente.
###############################################################################


###############################################################################
# 1) IMPORT E LOG INIZIALE
###############################################################################
print("importazione librerie...")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

print("caricamento datasetcompleto...")

# --- SPIEGAZIONE MINUZIOSA ---
# print("importazione librerie...")
# - Messaggio di log: utile per capire dove sei quando esegui lo script.

# import torch
# - PyTorch è il backend di training (tensori, GPU, autograd).
# - In questo script non lo usi direttamente, ma Transformers lo usa sotto.

# from transformers import ...
# - GPT2LMHeadModel:
#   - architettura GPT-2 + “LM Head” (strato finale) per predire distribuzione sul vocabolario.
# - GPT2Tokenizer:
#   - converte testo -> token IDs e viceversa.
# - Trainer:
#   - classe high-level che gestisce training loop, eval, logging, saving, ecc.
# - TrainingArguments:
#   - oggetto di configurazione: batch size, epoche, output_dir, ecc.

# from datasets import load_dataset
# - HuggingFace Datasets: download e gestione dataset in modo standard.
# - load_dataset("wikitext", ...) scarica e prepara splits (train/validation/test).

# print("caricamento datasetcompleto...")
# - Solo log; il nome è un po’ confuso (non carichi ancora, lo fai dopo).


###############################################################################
# 2) DOWNLOAD E PREPARAZIONE DATASET (WikiText-2)
###############################################################################
print("\nDownload dataset wikitext-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
small_train = dataset["train"].select(range(500)) 
small_eval = dataset["validation"].select(range(100)) 
print("Dataset wikitext-2 scaricato e suddiviso in train e validation.")
print(f"Dataset di esempio:{len(small_train)} righe per il training e {len(small_eval)} righe per la validazione.")

# --- SPIEGAZIONE MINUZIOSA ---
# print("\nDownload dataset wikitext-2...")
# - \n inserisce una riga vuota prima: migliora la leggibilità dei log.

# dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
# - Scarica/usa cache locale del dataset:
#   - "wikitext" è il nome dataset.
#   - "wikitext-2-raw-v1" è la configurazione:
#     - “raw” significa testo “grezzo” (include newline e formattazioni).
# - Ritorna un DatasetDict, tipicamente con chiavi:
#   - dataset["train"], dataset["validation"], dataset["test"]

# small_train = dataset["train"].select(range(500))
# - Prendi solo le prime 500 righe per training.
# - Perché farlo?
#   - riduci tempo di training (demo didattica).
#   - utile su PC senza grande GPU.
# - Nota: “prime 500” NON è random:
#   - potresti introdurre bias (sempre la stessa porzione del dataset).
#   - in esperimenti seri: meglio shuffle prima (dataset["train"].shuffle(seed=...)).

# small_eval = dataset["validation"].select(range(100))
# - Stesso discorso, ma per validation (100 righe).

# print(f"... {len(small_train)} ... {len(small_eval)} ...")
# - Stampa dimensioni effettive: sanity check.
# - Nota estetica: manca uno spazio dopo “esempio:”.


###############################################################################
# 3) CARICAMENTO TOKENIZER E MODELLO GPT-2
###############################################################################
print("\nCaricamento tokenizer e modello GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Tokenizer e modello GPT-2 caricati.")

# --- SPIEGAZIONE MINUZIOSA ---
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# - Scarica (o usa cache) del tokenizer GPT-2.
# - Il tokenizer GPT-2 usa byte-level BPE:
#   - gestisce bene qualsiasi stringa (anche caratteri strani)
#   - spezza in subword/token.

# tokenizer.pad_token = tokenizer.eos_token
# - GPT-2 “storicamente” non ha un token di padding dedicato.
# - Però, quando fai batching con padding="max_length", servirebbe un pad_token.
# - Soluzione comune: usare eos_token (end-of-sequence) come pad_token.
#
# Perché funziona?
# - A livello di forma, ti permette di riempire sequenze più corte fino a max_length.
# - Però attenzione concettuale:
#   - eos significa “fine frase”; pad significa “riempitivo”.
#   - Se non mascheri le posizioni padded nelle labels, il modello impara anche su padding,
#     cosa che può peggiorare il training.
# - Il Trainer / data collator spesso maschera automaticamente, ma nel tuo caso
#   stai creando labels = input_ids senza ignorare pad: questo è un punto delicato.

# model = GPT2LMHeadModel.from_pretrained("gpt2")
# - Carica pesi pre-addestrati GPT-2 (base).
# - LMHeadModel include lo strato finale per predire token successivo.
#
# Nota:
# - di default il modello viene caricato su CPU.
# - Trainer lo sposterà su GPU automaticamente se disponibile (in molti setup),
#   ma dipende dall’ambiente (e da accelerate, ecc).


###############################################################################
# 4) PREPARAZIONE DATI: TOKENIZZAZIONE + LABELS
###############################################################################
print("preparazione dati...")

def prepare_data(examples):
    inputs = tokenizer(examples["text"],
             truncation=True, 
             padding="max_length", 
             max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

train_data = small_train.map(prepare_data, batched=True)
eval_data = small_eval.map(prepare_data, batched=True)
print("Dati preparati per il training.")

# --- SPIEGAZIONE MINUZIOSA ---
# print("preparazione dati...")
# - Log.

# def prepare_data(examples):
# - Funzione che trasforma esempi grezzi del dataset in features per training.

# inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
# - examples["text"] è una lista di stringhe quando batched=True.
# - tokenizer(...) restituisce un dizionario con chiavi tipiche:
#   - "input_ids": lista di liste (token ids)
#   - "attention_mask": 1 dove c’è token reale, 0 dove c’è padding
#
# truncation=True
# - Se il testo supera max_length, taglia.
# - Motivo:
#   - semplifica batching (tutti 128 token).
#   - riduce memoria/tempo.
# - Contro:
#   - perdi informazione oltre i 128 token.
#   - per language modeling serio, spesso si usano blocchi concatenati di testo.

# padding="max_length"
# - Tutte le sequenze diventano esattamente lunghe max_length.
# - Motivo:
#   - batching semplice e prevedibile.
# - Contro:
#   - sprechi compute su padding se molte righe sono corte.
#   - alternativa: padding dinamico (“longest”) per batch.

# max_length=128
# - Lunghezza fissa. 128 è una scelta “leggera” per demo.
# - GPT-2 supporta contesti più lunghi (dipende dal modello, spesso 1024 token),
#   ma aumenterebbe costi di training.

# inputs["labels"] = inputs["input_ids"].copy()
# - Per causal language modeling:
#   - l’obiettivo è predire ogni token successivo.
#   - Transformers si aspetta "labels" con stessi token ids.
# - Il modello internamente fa lo shift:
#   - input: token_0..token_{n-1}
#   - labels: token_0..token_{n-1}
#   - la loss confronta predizione di token_t data history <t.
#
# Perché .copy()?
# - inputs["input_ids"] è una lista (o array-like).
# - Copi per evitare riferimenti condivisi:
#   - se qualcuno modificasse input_ids, non vuoi cambiare anche labels.
#
# PROBLEMA DIDATTICO IMPORTANTE:
# - Se fai padding con eos_token e poi metti labels = input_ids, stai dicendo al modello:
#   “impara anche a predire quei token di padding”.
# - In training serio, le posizioni di padding nelle labels si mettono a -100
#   (valore speciale ignorato dalla loss CrossEntropy in Transformers).
# - Qui non lo fai: può degradare la qualità, ma per demo breve può “funzionare”.

# return inputs
# - Restituisci un dict con almeno input_ids/attention_mask/labels.

# train_data = small_train.map(prepare_data, batched=True)
# - .map applica la funzione a tutto il dataset.
# - batched=True:
#   - examples è un batch (dict di liste) e tokenizer può essere più efficiente.
# - Risultato: train_data contiene nuove colonne (input_ids, attention_mask, labels).

# eval_data = small_eval.map(...)
# - Stesso per validation.

# print("Dati preparati ...")
# - Log.


###############################################################################
# 5) CONFIGURAZIONE TRAINING (TrainingArguments)
###############################################################################
print("\nConfigurazione del training...")

training_args = TrainingArguments(
    output_dir="./risultati",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=500,
    warmup_steps=100,
    logging_steps=50
    )

# --- SPIEGAZIONE MINUZIOSA ---
# TrainingArguments(...) raccoglie parametri del training loop.

# output_dir="./risultati"
# - Cartella dove salvare checkpoint, logs, ecc.
# - Se non esiste, viene creata.

# num_train_epochs=1
# - Numero di passate complete sul training set.
# - 1 epoca qui è demo: addestramento minimo.

# per_device_train_batch_size=4
# - Batch size per GPU/CPU device.
# - Se hai 1 GPU, batch=4 globale.
# - Se hai più GPU e distributed, il batch globale cresce.

# per_device_eval_batch_size=4
# - Batch size in evaluation.

# eval_strategy="steps"
# - ATTENZIONE: nelle versioni moderne di Transformers spesso si chiama
#   "evaluation_strategy" e non "eval_strategy".
# - Se la tua versione accetta eval_strategy, ok.
# - Se no, potresti avere errore TypeError: got an unexpected keyword argument.
#
# Significato:
# - fai evaluation ogni N steps (invece che a fine epoca).

# eval_steps=100
# - Ogni 100 step di training, esegui evaluation su eval_dataset.
# - Con dataset piccolo, potresti fare pochissime eval o anche nessuna,
#   dipende da quanti step totali fai.

# save_steps=500
# - Salva checkpoint ogni 500 steps.
# - Con training piccolo e 1 epoca, potresti non arrivare a 500 step:
#   quindi nessun checkpoint intermedio.

# warmup_steps=100
# - Warmup learning rate:
#   - LR cresce gradualmente nei primi 100 step.
# - Motivo:
#   - stabilizza training e evita gradienti troppo aggressivi all’inizio.

# logging_steps=50
# - Ogni 50 step logga metriche (loss ecc).


###############################################################################
# 6) CREAZIONE TRAINER
###############################################################################
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer
)
print("Inizio del training...")
trainer.train()
print("Training completato.")

# --- SPIEGAZIONE MINUZIOSA ---
# trainer = Trainer(...)
# - Incarna il training loop completo.
# - Parametri:
#   model=model:
#     - GPT2LMHeadModel (causal LM)
#   args=training_args:
#     - tutte le impostazioni
#   train_dataset=train_data:
#     - dataset tokenizzato con labels
#   eval_dataset=eval_data:
#     - dataset validazione tokenizzato
#   tokenizer=tokenizer:
#     - usato per salvataggio, padding, e alcune utilità.
#
# Nota:
# - Qui NON specifichi un data_collator.
# - Per causal LM spesso si usa DataCollatorForLanguageModeling o simili.
# - Senza data_collator specifico:
#   - Trainer userà un collator di default.
#   - Dal momento che tu hai già padding a max_length,
#     dovrebbe “impacchettare” batch senza problemi.
#
# print("Inizio del training...")
# - log.
#
# trainer.train()
# - Avvia training:
#   - forward pass, loss, backward, optimizer step, scheduler step, logging, ecc.
#
# print("Training completato.")
# - log.


###############################################################################
# 7) SALVATAGGIO MODELLO E TOKENIZER FINE-TUNED
###############################################################################
print("\nSalvataggio del modello fine-tuned...")
model.save_pretrained("./gpt2-mio")
tokenizer.save_pretrained("./gpt2-mio")
print("Modello fine-tuned salvato nella directory './gpt2-mio'.")

# --- SPIEGAZIONE MINUZIOSA ---
# model.save_pretrained("./gpt2-mio")
# - Salva pesi e config in una cartella in formato HuggingFace standard.
# - Questo permette di ricaricare con from_pretrained("./gpt2-mio").
#
# tokenizer.save_pretrained("./gpt2-mio")
# - Salva i file del tokenizer (vocab, merges, config).
# - Importante: modello e tokenizer devono rimanere “accoppiati”.
#
# Nota:
# - Stai salvando il modello “in memoria” dopo training.
# - Se avessi checkpoint intermedi, potresti anche voler salvare l’ultimo best model.


###############################################################################
# 8) GENERAZIONE TESTO CON IL MODELLO FINE-TUNED
###############################################################################
print("\nEsempio di generazione testo con il modello fine-tuned...")
from transformers import pipeline
generator = pipeline("text-generation", model="./gpt2-mio")
prompt = [
    "artificial intelligence is transforming the world by",
    "in the future, we will see more advancements in technology such as"
]

print("Generazione testo per i prompt:")
for p in prompt:
    result = generator(
        p, max_length=50, 
        num_return_sequences=1, 
        temperature=0.7, 
        do_sample=True
    )
    print(f"\nPrompt: {p}\nGenerated Text: {result[0]['generated_text']}")

# --- SPIEGAZIONE MINUZIOSA ---
# from transformers import pipeline
# - pipeline è un wrapper “alto livello” per inferenza veloce.
# - Ti evita di scrivere tokenization + generate() manualmente.

# generator = pipeline("text-generation", model="./gpt2-mio")
# - Crea una pipeline per generazione.
# - model="./gpt2-mio" indica che deve caricare il modello dalla cartella salvata.
# - NOTA: non passi esplicitamente tokenizer:
#   - spesso pipeline lo carica automaticamente dalla stessa cartella.
#   - ma in alcuni casi è meglio specificare anche tokenizer="./gpt2-mio".

# prompt = [ ... ]
# - Lista di prompt su cui vuoi generare continuazioni.
# - Nota: i prompt sono in inglese, coerenti con WikiText (inglese).

# for p in prompt:
# - iteri sui prompt.

# result = generator(p, max_length=50, num_return_sequences=1, temperature=0.7, do_sample=True)
# Parametri generazione:
# - max_length=50:
#   - lunghezza totale della sequenza (prompt + continuazione) in token.
#   - quindi la parte generata è “fino a” arrivare a 50 token totali.
#
# - num_return_sequences=1:
#   - genera una sola continuazione per prompt.
#   - se metti 3, ottieni 3 varianti.
#
# - temperature=0.7:
#   - controlla casualità:
#     - <1 rende più “conservativo”
#     - >1 più creativo/variabile
#
# - do_sample=True:
#   - abilita sampling (scelta probabilistica dei token).
#   - Se fosse False, spesso usa greedy decoding (deterministico).
#
# result è tipicamente una lista di dict, es:
#   [{"generated_text": "..."}]
#
# print(f"... {result[0]['generated_text']}")
# - Stampa il testo generato (prima sequenza).


###############################################################################
# NOTE “DA PROF” (sulle scelte di coding più importanti)
###############################################################################
# 1) Padding + labels:
#    - Il punto più delicato è: padding con eos_token e labels=input_ids.
#    - In training serio, si mascherano i pad token nella loss usando -100 nelle labels.
#
# 2) eval_strategy vs evaluation_strategy:
#    - dipende dalla versione Transformers.
#    - se ti esplode, è quasi sicuramente quello.
#
# 3) Data collator:
#    - spesso si usa un collator dedicato per causal LM.
#    - qui hai già padding fisso, quindi può funzionare lo stesso.
#
# 4) Subset non random:
#    - select(range(500)) prende sempre le prime righe.
#    - per esperimenti seri: shuffle.
###############################################################################
