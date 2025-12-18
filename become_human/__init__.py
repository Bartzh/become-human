import os
from dotenv import load_dotenv
from langchain_dev_utils.chat_models import register_model_provider
from langchain_dev_utils.embeddings import register_embeddings_provider

load_dotenv()

if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./config"):
    os.makedirs("./config")

register_model_provider(
    provider_name='openai',
    chat_model='openai-compatible'
)

register_embeddings_provider(
    provider_name='openai',
    embeddings_model='openai-compatible'
)
