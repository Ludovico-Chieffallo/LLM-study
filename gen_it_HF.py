from transformers import pipeline

model= "groNLP/gpt2-small-italian"
generator = pipeline("text-generation", model = model)
prompt = "In un futuro lontano, l'umanità ha"
print(generator(prompt, max_new_tokens=100, num_return_sequences=1)[0],["generated_text"])