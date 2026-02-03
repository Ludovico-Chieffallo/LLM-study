from transformers import pipeline

sentiment= pipeline("sentiment-analysis")
print(sentiment("I love using Hugging Face transformers library!"))
print(sentiment("I hate waiting in long lines."))