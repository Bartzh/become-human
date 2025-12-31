import os
from dotenv import load_dotenv
from langchain_dev_utils.chat_models import batch_register_model_provider
from langchain_dev_utils.embeddings import batch_register_embeddings_provider
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./config"):
    os.makedirs("./config")

batch_register_model_provider([
    {
        'provider_name': 'openai',
        'chat_model': 'openai-compatible'
    },
    {
        'provider_name': 'dashscope',
        'chat_model': 'openai-compatible'
    },
    {
        'provider_name': 'openrouter',
        'chat_model': 'openai-compatible'
    },
    {
        'provider_name': 'anthropic',
        'chat_model': ChatAnthropic
    },
    {
        'provider_name': 'ollama',
        'chat_model': ChatOllama
    }
])

batch_register_embeddings_provider([
    {
        'provider_name': 'openai',
        'embeddings_model': 'openai-compatible'
    },
    {
        'provider_name': 'openrouter',
        'embeddings_model': 'openai-compatible'
    },
    {
        'provider_name': 'ollama',
        'embeddings_model': OllamaEmbeddings
    }
])
