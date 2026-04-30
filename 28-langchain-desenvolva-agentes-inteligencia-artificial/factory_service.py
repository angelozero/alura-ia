import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings

# Load environment variables from .env file
load_dotenv()

# Configuration constants retrieved from environment
MODEL_NAME = os.getenv("MODEL_NAME")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


def get_chat_model(model_name=MODEL_NAME):
    """
    Configuração baseada na imagem do VSCode/LiteLLM.
    """
    return init_chat_model(
        model=model_name,            # "gemini-2.0-flash"
        model_provider="openai",     # O Proxy do LiteLLM usa o padrão OpenAI
        api_key=API_KEY,             # Sua chave secreta
        base_url=BASE_URL,           # A URL do Flow CI&T
        temperature=0,
    )


def get_embeddings(model_name=OLLAMA_MODEL_NAME):
    """
    Initializes and returns the Embeddings instance.
    Configured to handle float encoding format for Ollama compatibility.
    """
    return OllamaEmbeddings(
        model=model_name,
        base_url="http://localhost:11434" # URL padrão do Ollama
    )