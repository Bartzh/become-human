import os, sys
from typing import TYPE_CHECKING
from dotenv import load_dotenv
from loguru import logger
from langchain_dev_utils.chat_models import batch_register_model_provider
from langchain_dev_utils.embeddings import batch_register_embeddings_provider
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings

if TYPE_CHECKING:
    from become_human.manager import sprite_manager

__all__ = ['sprite_manager']

def __getattr__(name):
    if name == 'sprite_manager':
        from become_human.manager import sprite_manager
        return sprite_manager
    raise AttributeError(f"module {__name__} has no attribute {name}")

load_dotenv()

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stdout, level=log_level)
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="2 weeks",
    enqueue=True,
    level=log_level
)

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
