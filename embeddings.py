"""
Module for generating text embeddings using Ollama API.
"""

import requests
import json
import yaml
import os
import numpy as np
import logging
from tqdm import tqdm

# Load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

EMBEDDINGS_MODEL = config['models']['embeddings']['name']
API_URL = "http://localhost:11434/api"

# Set up logging
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up logging for embeddings operations
logging.basicConfig(
    filename='logs/embeddings.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_ollama_setup(model, debug=False):
    """
    Validate that Ollama is running and the requested model is available.
    Download the model if it's not already available.
    
    Args:
        model (str): The model name to validate
        
    Returns:
        bool: True if validation passed
        
    Raises:
        Exception: If Ollama is not running or model couldn't be loaded
    """
    # First check if Ollama is running
    if not test_ollama_connection():
        raise Exception("Ollama is not running. Please start the Ollama service.")
    
    # Then check if the model is available
    try:
        response = requests.get(f"{API_URL}/tags", timeout=10)
        response.raise_for_status()
        available_models = [model['name'] for model in response.json().get('models', [])]
        
        if model in available_models:
            if debug:
                logging.info(f"Model {model} is available.")
            return True
        else:
            logging.info(f"Model {model} not found. Pulling model...")
            print(f"Model {model} not found. Pulling model...")
            
            # Pull the model
            pull_response = requests.post(
                f"{API_URL}/pull",
                json={"name": model},
                timeout=600  # Longer timeout for model download
            )
            pull_response.raise_for_status()
            logging.info(f"Successfully pulled model {model}")
            return True
            
    except Exception as e:
        logging.error(f"Error validating Ollama setup: {e}")
        raise Exception(f"Error validating Ollama setup: {e}")

def get_embeddings(text, model=None):
    """
    Generate embeddings for a given text using Ollama.
    
    Args:
        text (str): The text to generate embeddings for
        model (str, optional): The model to use. Defaults to the one in config.
        
    Returns:
        list: The embedding vector or None if there was an error
    """
    if model is None:
        model = EMBEDDINGS_MODEL
        
    # Remove the ollama/ prefix if present (API doesn't need it)
    if model.startswith("ollama/"):
        model = model.replace("ollama/", "", 1)
    
    try:
        validate_ollama_setup(model)
        
        response = requests.post(
            f"{API_URL}/embeddings",
            json={
                "model": model,
                "prompt": text,
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("embedding")
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None

def get_embeddings_batch(texts, model=None, batch_size=10):
    """
    Generate embeddings for a batch of texts.
    
    Args:
        texts (list): List of texts to generate embeddings for
        model (str, optional): The model to use
        batch_size (int): Number of texts to process in each batch
        
    Returns:
        list: List of embedding vectors
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    logging.info(f"Starting batch embedding generation for {len(texts)} texts")
    
    # Use tqdm for the batches with minimal output
    batch_pbar = tqdm(
        total=total_batches, 
        desc="Processing batches", 
        unit="batch", 
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_num = i//batch_size + 1
        logging.info(f"Processing batch {batch_num}/{total_batches}")
        
        # Use a nested tqdm for items within the batch
        for text in batch:
            embedding = get_embeddings(text, model)
            all_embeddings.append(embedding)
        
        batch_pbar.update(1)
    
    batch_pbar.close()
    logging.info(f"Completed batch embedding generation for {len(texts)} texts")
    
    return all_embeddings

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1 (list): First vector
        vec2 (list): Second vector
        
    Returns:
        float: Cosine similarity value between -1 and 1
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0
        
    return dot_product / (norm_a * norm_b)

def test_ollama_connection():
    """
    Test if Ollama server is running and available.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        response = requests.get(f"{API_URL}/tags", timeout=5)
        response.raise_for_status()
        logging.info("Successfully connected to Ollama API")
        return True
    except Exception as e:
        logging.error(f"Could not connect to Ollama: {e}")
        return False

if __name__ == "__main__":
    # Test the embedding functionality
    if test_ollama_connection():
        print(f"Connected to Ollama API. Using model: {EMBEDDINGS_MODEL}")
        
        test_text = "This is a test sentence to check if embeddings are working correctly."
        embeddings = get_embeddings(test_text)
        
        if embeddings:
            print(f"Successfully generated embeddings. Vector dimension: {len(embeddings)}")
            logging.info(f"Test embeddings generated. Vector dimension: {len(embeddings)}")
        else:
            print("Failed to generate embeddings.")
            logging.error("Failed to generate test embeddings.")
    else:
        print("Ollama connection failed. Please ensure Ollama is running.")
        logging.error("Ollama connection failed during test.") 