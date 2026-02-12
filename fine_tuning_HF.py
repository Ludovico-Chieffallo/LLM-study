print("importazione librerie...")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

print("caricamento datasetcompleto...")

print("\nDownload dataset wikitext-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
small_train = dataset["train"].select(range(500)) 
small_eval = dataset["validation"].select(range(100)) 
print("Dataset wikitext-2 scaricato e suddiviso in train e validation.")
print(f"Dataset di esempio:{len(small_train)} righe per il training e {len(small_eval)} righe per la validazione.")

print("\nCaricamento tokenizer e modello GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Tokenizer e modello GPT-2 caricati.")

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

print("\nSalvataggio del modello fine-tuned...")
model.save_pretrained("./gpt2-mio")
tokenizer.save_pretrained("./gpt2-mio")
print("Modello fine-tuned salvato nella directory './gpt2-mio'.")

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