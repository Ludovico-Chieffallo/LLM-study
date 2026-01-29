from transformers import pipeline
import logging
import torch


try:
    generator = pipeline('text-generation', model='gpt2')
    print("pipeline loaded successfully.")
except ImportError as e:
    print("ImportError:", e)
    generator = None
except Exception as e:
    print("An error occurred while loading the pipeline:", e)
    generator = None


