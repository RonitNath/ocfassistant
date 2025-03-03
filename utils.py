import os
import shutil
from dotenv import load_dotenv

def setup_env():
    """
    If .env doesn't exist, copy .env.template to .env and fill in the values.
    """
    if not os.path.exists(".env"):
        shutil.copy(".env.template", ".env")
        print("Copied .env.template default values to .env")

    # Load the environment variables
    load_dotenv()

    # Get the environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    ollama_url = os.getenv("OLLAMA_URL")

    return qdrant_url, ollama_url
