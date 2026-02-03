from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-it")
print(translator("Hello, how are you?")[0]["translation_text"])
