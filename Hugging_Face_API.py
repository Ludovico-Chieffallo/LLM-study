from transformers import pipeline, GenerationConfig
import logging
import torch


try:
    generator = pipeline('text-generation', model="gpt2")
    print("pipeline loaded successfully.")
except ImportError as e:
    print("ImportError:", e)
    generator = None
except Exception as e:
    print("An error occurred while loading the pipeline:", e)
    generator = None 

torch.manual_seed(44)
if generator:
    prompt = "I love life because "
    print(f"Generating text for prompt: '{prompt}'")
    try:
        gen_cfg = GenerationConfig(
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=generator.tokenizer.eos_token_id
        )

        response = generator(prompt, generation_config=gen_cfg)
        print("\n -- Results -- ")
        print("complete tx: ", response)
        if response:
            text_complete= response[0]['generated_text']
            print("Generated Text: ", text_complete)
            new_text = text_complete[len(prompt):]
            print("\nNewly Generated Text: ")
            print(new_text.strip())
            print("-----------\n")

        else:
            print("No response received from the generator.")

    except Exception as e:
        print("An error occurred during text generation:", e)
else:
    print("Generator pipeline is not available.")


