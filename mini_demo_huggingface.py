from transformers import pipeline

generator = pipeline("text-generation", model = "gpt2") #possiamo usare anche sentiment-analysis, question-answering, transtion_en_to_it, etc. (cambiando modello)
prompt = "In a distant future, humanity has"
print(f"Generating text for prompt: '{prompt}'")    
response = generator(prompt, max_new_tokens=100, num_return_sequences=1)
print("\n -- Results -- ")
print("complete tx: ", response)