#it works only if you have transformers 4.44.2 library installed
from transformers import pipeline

translator = pipeline("translation_en_to_it", model="Helsinki-NLP/opus-mt-en-it")
print(translator("Hello, how are you?")[0]["translation_text"])