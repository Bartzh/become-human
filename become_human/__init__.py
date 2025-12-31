import os
from dotenv import load_dotenv
from langchain_dev_utils.chat_models import batch_register_model_provider
from langchain_dev_utils.embeddings import batch_register_embeddings_provider
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

provider_names = [
    'openai',
    'dashscope',
    'openrouter',
    'anthropic',
    'ollama'
]
model_providers = {}
embeddings_providers = {}
for name in provider_names:
    if os.getenv(f"{name.upper()}_API_BASE"):
        if name in ['openai', 'openrouter', 'dashscope']:
            model_providers[name] = 'openai-compatible'
            if name != 'dashscope':
                embeddings_providers[name] = 'openai-compatible'
        elif name == 'anthropic':
            model_providers[name] = ChatAnthropic
        elif name == 'ollama':
            model_providers[name] = ChatOllama
            embeddings_providers[name] = OllamaEmbeddings

batch_register_model_provider([
    {'provider_name': n, 'chat_model': m}
    for n, m in model_providers.items()
])
batch_register_embeddings_provider([
    {'provider_name': n, 'embeddings_model': m}
    for n, m in embeddings_providers.items()
])

if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./config"):
    os.makedirs("./config")
