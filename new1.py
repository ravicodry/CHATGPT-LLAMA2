from transformers import AutoTokenizer
import transformers
import torch
import os

torch.cuda.empty_cache()
#from transformers import set_token
import streamlit as st
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