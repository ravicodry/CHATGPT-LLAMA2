from transformers import AutoTokenizer
import transformers
import torch
import os
from dotenv import load_dotenv
load_dotenv()
import subprocess
torch.cuda.empty_cache()
#from transformers import set_token
import streamlit as st

api_token = st.text_input("Enter your Hugging Face API Token:", type="password")

# Create a button to submit the token and run Hugging Face CLI
if st.button("Run Hugging Face CLI"):
    # Check if the user entered a token
    if api_token:
        # Set the HUGGINGFACE_TOKEN environment variable with the provided token
        # This allows the Hugging Face CLI to authenticate API requests
        subprocess.run(["transformers-cli", "login", f"--api_key {api_token}"])
        st.success("Hugging Face CLI is authenticated with the provided API Token.")
    else:
        st.warning("Please enter an API Token.")
st.title("Chatbot App")
user_input = st.text_input("")
# Set your Hugging Face API token
#set_token("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")
conversation = []
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
token="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"
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
if user_input:
  conversation.append(f"{user_input}")
  sequences = pipeline(
      " ".join(conversation),
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=50,
  )
  # for seq in sequences:
  #     print(f"Result: {seq['generated_text']}")
  st.write(sequences)