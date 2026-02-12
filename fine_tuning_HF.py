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