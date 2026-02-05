import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGING_FACEHUB_API_TOKEN")
if hf_token:
    os.environ["HUGGING_FACEHUB_API_TOKEN"] = hf_token
    print("Hugging Face API token found.")
else:
    print("Hugging Face API token not found. Please set the HUGGINGFACE_API_TOKEN environment variable.")

repo_id = "HuggingFaceH4/zephyr-7b_beta"
if repo_id:
    print(f"Using Hugging Face model repository: {repo_id}")
else:
    print("Model repository ID not found. Please set the repo_id variable.")

if not os.getenv("HUGGING_FACEHUB_API_TOKEN"):
    print("Warning: Hugging Face API token is not set. API calls may fail.")
    llm_hf=None
else:
    llm_hf = HuggingFaceEndpoint(repo_id=repo_id, 
                                 task="text-generation",
                                 max_new_tokens=512,
                                 temperature=0.7,
                                 do_sample=True
                                 )
    print(f"Initialized Hugging Face LLM with repository: {repo_id}") #print

    