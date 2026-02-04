# Import the AutoTokenizer class from Hugging Face Transformers.
# This class automatically selects the correct tokenizer
# architecture based on the model name you provide (e.g., GPT-2, BERT, LLaMA).
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import PyTorch, which is the deep learning framework used
# to store tensors, run computations, and execute the neural network.
import torch


# Simple debug/log message to confirm that the imports worked correctly.
print("classes imported successfully")


# The identifier of the model we want to load from Hugging Face Hub.
# "gpt2" refers to the small GPT-2 language model released by OpenAI.
# This string is used by Hugging Face to locate:
#  - the model architecture configuration
#  - the pretrained weight files
#  - the tokenizer vocabulary and rules
model_name = "gpt2"


# Load the tokenizer associated with the specified model.
# from_pretrained() downloads (if necessary) and reconstructs
# the exact tokenizer used when the model was originally trained.
#
# The tokenizer is responsible for:
#  - splitting raw text into subword tokens
#  - mapping tokens to integer IDs
#  - creating attention masks
#  - adding special tokens if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Print visual separators and status messages to make
# the program output easier to read when running the script.
print("----------------------")
print(f"Tokenizer for {model_name} loaded successfully")
print("----------------------")


# Load the pretrained language model itself.
# AutoModelForCausalLM:
#  - chooses the correct GPT-style architecture
#  - loads the neural network weights
#  - prepares the model for text generation tasks
#
# This does NOT retrain the model; it only loads existing parameters.
model = AutoModelForCausalLM.from_pretrained(model_name)


# Log that the model was loaded correctly.
print("----------------------")
print(f"Model for {model_name} loaded successfully")
print("----------------------")


# Decide which hardware device should be used to run the model.
#
# torch.cuda.is_available():
#   returns True if PyTorch detects an NVIDIA GPU with CUDA support.
#
# If CUDA is available, we select "cuda".
# Otherwise, we fall back to "cpu".
#
# NOTE: On Apple Silicon Macs, this logic usually selects "cpu".
# To use the Apple GPU, one would typically check for "mps" instead.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Move all model parameters (weights and internal buffers)
# to the selected device.
# If the device is "cuda", the model will run on the GPU.
# If it is "cpu", all computations will happen on the processor.
model.to(device)


# Print which device is being used.
print("----------------------")
print(f"Model moved to device: {device}")
print("----------------------")


# The text prompt that we want the model to continue.
# This is the initial sequence of tokens fed into the model
# for autoregressive text generation.
prompt = "Once upon a time"


# Convert the raw text prompt into PyTorch tensors.
#
# return_tensors="pt" means:
#   - return PyTorch tensors instead of Python lists or NumPy arrays.
#
# The tokenizer produces a dictionary that typically contains:
#   - "input_ids": numerical token IDs
#   - "attention_mask": 1s and 0s indicating which tokens are real
#     and which are padding (if any).
inputs = tokenizer(prompt, return_tensors="pt")  # "pt" stands for PyTorch


# Print the original input text for reference.
print("input_text", prompt)


# Display separators and the full tokenized output.
print("----------------------")
print("Tokenized inputs:", inputs)
print("----------------------")


# Print the actual tensor of token IDs.
# Each integer corresponds to a token in the model's vocabulary.
print("input ids:", inputs['input_ids'])


# Print the attention mask tensor.
# A value of 1 means the token should be attended to.
# A value of 0 means it is padding and should be ignored.
print("attention mask:", inputs['attention_mask'])


# Move every tensor inside the inputs dictionary
# to the same device as the model.
#
# This is absolutely necessary:
# PyTorch requires that the model and all input tensors
# live on the same device, otherwise it raises an error.
inputs = {key: value.to(device) for key, value in inputs.items()}


# Log that the inputs were successfully moved.
print("----------------------")
print("Inputs moved to device:", device)
print("----------------------")


# Generate new tokens using the model.
#
# model.generate() runs autoregressive decoding:
#   - it repeatedly predicts the next token
#   - appends it to the sequence
#   - feeds the updated sequence back into the model
#
# Parameters:
# --------------------------------------------------
# input_ids:
#   the initial token IDs to start generation from.
#
# attention_mask:
#   tells the model which tokens are valid.
#
# max_new_tokens:
#   maximum number of new tokens to generate beyond the prompt.
#
# num_return_sequences:
#   how many different generated sequences to produce.
#
# no_repeat_ngram_size:
#   prevents the model from repeating any n-gram
#   (here: sequences of 2 tokens) to reduce looping.
#
# pad_token_id:
#   token ID used for padding if sequences need to be aligned.
#   GPT-2 does not have a native pad token, so we reuse EOS.
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.eos_token_id
)


# Print the raw output tensor.
# This is a 2D tensor of shape:
#   (num_return_sequences, sequence_length)
# containing token IDs for the full generated text.
print("----------------------")
print("Generated outputs:", outputs)
print("----------------------")


# Print the shape of the output tensor to confirm its dimensions.
print("output shape:", outputs.shape)


# Convert the generated token IDs back into human-readable text.
#
# outputs[0] selects the first generated sequence.
# skip_special_tokens=True removes tokens such as <eos> or <pad>.
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


# Print separators and the final generated text.
print("----------------------")
print("Generated text:", generated_text)
print("----------------------")
