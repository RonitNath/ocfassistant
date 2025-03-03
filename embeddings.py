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
import ollama
import time

# Load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# API endpoint from environment variable
API_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api"

# Set up logging
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up logging for embeddings operations
logging.basicConfig(
    filename='logs/embeddings.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get embeddings model from config
model = config['embeddings_model']
data_folder = config.get('data_folder', 'data')

def validate_ollama_setup(model):
    """
    Validates that Ollama is running and the specified model is available.
    Returns True if setup is valid, False otherwise.
    """
    try:
        # Check if Ollama is running by making a simple request
        response = requests.get(f"{API_URL}/version", timeout=5)
        if response.status_code != 200:
            logging.error(f"Ollama API returned status code {response.status_code}")
            print(f"Error: Ollama API returned status code {response.status_code}. Is Ollama running?")
            return False

        # Check if the model exists
        response = requests.get(f"{API_URL}/tags", timeout=5)
        if response.status_code != 200:
            logging.error(f"Failed to get model list: {response.status_code}")
            return False
            
        available_models = [model_info['name'] for model_info in response.json().get('models', [])]
        
        if model not in available_models:
            print(f"Model '{model}' not found in Ollama. Available models: {', '.join(available_models)}")
            print("\nTo install the model, run: ollama pull " + model)
            return False
            
        return True
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama API. Is Ollama running on http://localhost:11434?")
        logging.error("Connection to Ollama API failed.")
        return False
    except Exception as e:
        print(f"Error validating Ollama setup: {str(e)}")
        logging.error(f"Error validating Ollama setup: {str(e)}")
        return False

def get_embeddings(text, model=None):
    """
    Get embeddings for the provided text.
    
    Args:
        text (str): Text to generate embeddings for
        model (str): Embedding model to use, defaults to config
        
    Returns:
        np.ndarray: The embeddings for the text
    """
    if model is None:
        model = config['embeddings_model']
    
    # Validate Ollama setup
    if not validate_ollama_setup(model):
        raise RuntimeError("Ollama setup validation failed")
        
    # Call Ollama to generate embeddings
    url = f"{API_URL}/embeddings"
    data = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        embeddings = response.json().get("embedding", [])
        return np.array(embeddings, dtype=np.float32)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting embeddings: {str(e)}")
        raise RuntimeError(f"Failed to get embeddings: {str(e)}")

def get_chunk_embedding_path(chunk_path):
    """
    Get the path where the embedding for a chunk should be stored.
    
    Args:
        chunk_path (str): Path to the chunk file
        
    Returns:
        str: Path to the embedding file
    """
    # Replace .txt with .json for the embedding file
    embedding_path = chunk_path.replace('.txt', '.embedding.json')
    return embedding_path

def embedding_exists(chunk_path):
    """
    Check if an embedding exists for a chunk.
    
    Args:
        chunk_path (str): Path to the chunk file
        
    Returns:
        bool: True if embedding exists, False otherwise
    """
    embedding_path = get_chunk_embedding_path(chunk_path)
    return os.path.exists(embedding_path)

def save_embedding(chunk_path, embedding, metadata=None):
    """
    Save an embedding for a chunk.
    
    Args:
        chunk_path (str): Path to the chunk file
        embedding (numpy.ndarray): Embedding vector
        metadata (dict): Additional metadata to store with the embedding
        
    Returns:
        str: Path to the saved embedding file
    """
    embedding_path = get_chunk_embedding_path(chunk_path)
    
    # Create metadata if not provided
    if metadata is None:
        metadata = {}
    
    # Extract the folder path (which contains the content hash)
    folder_path = os.path.dirname(chunk_path)
    content_hash = os.path.basename(folder_path)
    
    # Add additional metadata
    metadata.update({
        'chunk_path': chunk_path,
        'content_hash': content_hash,
        'embedding_model': model,
        'timestamp': time.time(),
    })
    
    # Convert embedding to list for JSON serialization
    embedding_data = {
        'embedding': embedding.tolist(),
        'metadata': metadata
    }
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    
    # Save the embedding
    with open(embedding_path, 'w') as f:
        json.dump(embedding_data, f)
        
    return embedding_path

def load_embedding(chunk_path):
    """
    Load an embedding for a chunk.
    
    Args:
        chunk_path (str): Path to the chunk file
        
    Returns:
        tuple: (numpy.ndarray, dict) - The embedding vector and metadata
    """
    embedding_path = get_chunk_embedding_path(chunk_path)
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"No embedding found for {chunk_path}")
        
    with open(embedding_path, 'r') as f:
        data = json.load(f)
        
    embedding = np.array(data['embedding'])
    metadata = data.get('metadata', {})
    
    return embedding, metadata

def generate_embeddings_for_chunks(chunk_paths, force=False):
    """
    Generate embeddings for multiple chunks.
    
    Args:
        chunk_paths (list): List of paths to chunk files
        force (bool): Whether to regenerate embeddings even if they already exist
        
    Returns:
        dict: Mapping of chunk paths to embedding paths
    """
    # Filter out chunks that already have embeddings (unless force=True)
    if not force:
        paths_to_process = [path for path in chunk_paths if not embedding_exists(path)]
    else:
        paths_to_process = chunk_paths
    
    if not paths_to_process:
        print("No new embeddings to generate")
        return {}
    
    results = {}
    
    # Read all chunk texts
    chunk_texts = []
    for path in tqdm(paths_to_process, desc="Reading chunks", unit="chunk"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                chunk_texts.append(f.read())
        except Exception as e:
            logging.error(f"Error reading chunk {path}: {e}")
            continue
    
    # Generate embeddings in batch
    print(f"Generating embeddings for {len(chunk_texts)} chunks...")
    start_time = time.time()
    
    try:
        embeddings = get_embeddings(chunk_texts, batch=True)
        
        # Save each embedding with its metadata
        for i, (path, embedding) in enumerate(zip(paths_to_process, embeddings)):
            try:
                # Get the folder name (sanitized URL with hash)
                folder_name = os.path.basename(os.path.dirname(path))
                
                # Get the chunk index from the filename
                chunk_filename = os.path.basename(path)
                chunk_idx = int(chunk_filename.split('_')[1].split('.')[0]) if '_' in chunk_filename else 0
                
                # Create metadata
                metadata = {
                    'folder_name': folder_name,
                    'chunk_index': chunk_idx,
                    'chunk_file': chunk_filename
                }
                
                # Save the embedding
                embedding_path = save_embedding(path, embedding, metadata)
                results[path] = embedding_path
            except Exception as e:
                logging.error(f"Error saving embedding for {path}: {e}")
                continue
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise
    
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds")
    
    return results

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
    Test if Ollama API is available and running.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        # Try to connect to Ollama API
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/version", timeout=3)
        
        if response.status_code == 200:
            print(f"Successfully connected to Ollama API at {ollama_url}")
            return True
        else:
            print(f"Ollama API returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Ollama connection error: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error testing Ollama connection: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the embedding functionality
    if test_ollama_connection():
        print(f"Connected to Ollama API. Using model: {model}")
        
        test_text = "This is a test sentence to check if embeddings are working correctly."
        embeddings = get_embeddings(test_text)
        
        if embeddings is not None:
            print(f"Successfully generated embeddings. Vector dimension: {len(embeddings)}")
            logging.info(f"Test embeddings generated. Vector dimension: {len(embeddings)}")
        else:
            print("Failed to generate embeddings.")
            logging.error("Failed to generate test embeddings.")
    else:
        print("Ollama connection failed. Please ensure Ollama is running.")
        logging.error("Ollama connection failed during test.") 