"""
Module for generating text embeddings using Ollama API.
"""

import requests
import json
import yaml
import os
import numpy as np

# Load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

EMBEDDINGS_MODEL = config['models']['embeddings']['name']
API_URL = "http://localhost:11434/api"

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
                print(f"Model {model} is available.")
            return True
        else:
            print(f"Model {model} not found. Available models: {available_models}")
            print(f"Pulling model {model}...")
            
            # Pull the model
            pull_response = requests.post(
                f"{API_URL}/pull",
                json={"name": model},
                timeout=600  # Longer timeout for model download
            )
            pull_response.raise_for_status()
            return True
            
    except Exception as e:
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
        print(f"Error generating embeddings: {e}")
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
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
        
        for text in batch:
            embedding = get_embeddings(text, model)
            all_embeddings.append(embedding)
            
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
        return True
    except Exception as e:
        print(f"Could not connect to Ollama: {e}")
        return False

if __name__ == "__main__":
    # Test the embedding functionality
    if test_ollama_connection():
        print(f"Connected to Ollama API. Using model: {EMBEDDINGS_MODEL}")
        
        test_text = "This is a test sentence to check if embeddings are working correctly."
        embeddings = get_embeddings(test_text)
        
        if embeddings:
            print(f"Successfully generated embeddings. Vector dimension: {len(embeddings)}")
            print(f"First few dimensions: {embeddings[:5]}")
        else:
            print("Failed to generate embeddings.")
    else:
        print("Ollama connection failed. Please ensure Ollama is running.") 