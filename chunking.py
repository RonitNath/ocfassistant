"""
Module for semantic chunking of text data.

This module provides functions to break down large texts into semantically meaningful chunks,
using a combination of approaches including:
1. Text boundary detection (headings, paragraphs)
2. Semantic similarity based chunking
3. Size-based chunking with overlap
"""

import os
import re
import json
import numpy as np
import logging
from bs4 import BeautifulSoup
import embeddings
import nltk
from tqdm import tqdm
import yaml
import datetime
import hashlib
import time

# Load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure logging
logging.basicConfig(
    filename='logs/chunking.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Make sure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Import sent_tokenize after ensuring punkt is downloaded
from nltk.tokenize import sent_tokenize

import nltk

def ensure_nltk_resource(resource_path, resource_name=None):
    """
    Checks for an NLTK resource and downloads it if not found.
    
    Parameters:
        resource_path (str): The internal NLTK path (e.g., 'tokenizers/punkt_tab').
        resource_name (str): The name to pass to nltk.download. If None,
                             the function derives it from the resource_path.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        if resource_name is None:
            # Derive resource name from the resource path; e.g., "punkt_tab" from "tokenizers/punkt_tab"
            resource_name = resource_path.split('/')[1]
        nltk.download(resource_name)
        # After downloading, you might want to verify the download succeeded:
        nltk.data.find(resource_path)

# Define a fallback sentence tokenizer in case NLTK's fails
def safe_sent_tokenize(text):
    """
    A safer version of sent_tokenize that falls back to simple splitting if NLTK fails.
    """
    try:
        return sent_tokenize(text)
    except Exception as e:
        print(f"NLTK tokenization failed: {e}")
        # Simple fallback using regex for periods followed by spaces and capital letters
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        if not sentences:
            # If all else fails, just return the text as a single sentence
            return [text]
        return sentences

# Configuration from config.yml
with open('config.yml', 'r') as f:
    from yaml import safe_load
    config = safe_load(f)

data_folder = config['data_folder']

# Chunking parameters with detailed explanations
# This parameter sets the maximum number of characters allowed in a single chunk. It ensures that chunks do not become too large and unwieldy for processing.
MAX_CHUNK_SIZE = config['chunking']['max_chunk_size']  
# This parameter sets the minimum number of characters required in a chunk. It prevents chunks from being too small and potentially losing context.
MIN_CHUNK_SIZE = config['chunking']['min_chunk_size']  
# This parameter determines the number of characters that should overlap between adjacent chunks. Overlapping chunks can help maintain context and ensure that important information is not lost at chunk boundaries.
OVERLAP_SIZE = config['chunking']['overlap_size']      
# This parameter sets the threshold for determining semantic similarity between sentences or chunks. It is used to decide whether two chunks should be merged or kept separate based on their semantic content.
SIMILARITY_THRESHOLD = config['chunking']['similarity_threshold']  

def chunk_text_file(file_path, output_dir=None, method="hybrid"):
    """
    Chunk a text file into semantically meaningful chunks.
    
    Args:
        file_path (str): Path to the text file
        output_dir (str): Directory to save the chunks (defaults to same directory as input)
        method (str): Chunking method to use: "size", "semantic", or "hybrid"
        
    Returns:
        list: List of chunks created
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    # Default output directory
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the text file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Skip empty files
    if not text.strip():
        print(f"Empty file: {file_path}")
        return []
    
    # Usage:
    ensure_nltk_resource('tokenizers/punkt_tab')

    # Choose chunking method
    if method == "size":
        chunks = chunk_by_size(text)
    elif method == "semantic":
        chunks = chunk_by_semantic(text)
    else:  # hybrid is default
        chunks = chunk_hybrid(text)
    
    # Save chunks to files
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(output_dir, f"{base_name}.chunk{i+1}.txt")
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(chunk)
    
    # Save chunk metadata
    metadata = {
        "original_file": file_path,
        "chunk_count": len(chunks),
        "chunk_method": method,
        "chunks": [
            {
                "index": i + 1,
                "size": len(chunk),
                "first_words": chunk[:50].replace("\n", " ").strip() + "..."
            } for i, chunk in enumerate(chunks)
        ]
    }
    
    metadata_file = os.path.join(output_dir, f"{base_name}.chunks.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return chunks

def chunk_by_size(text, max_size=MAX_CHUNK_SIZE, overlap=OVERLAP_SIZE):
    """
    Chunk text by fixed size with overlap, respecting sentence boundaries.
    
    Args:
        text (str): The text to chunk
        max_size (int): Maximum characters per chunk
        overlap (int): Number of characters to overlap
        
    Returns:
        list: List of text chunks
    """
    # Get sentences to preserve their boundaries
    sentences = safe_sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the max size, save the chunk and start a new one
        if len(current_chunk) + len(sentence) > max_size and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk)
            
            # Start new chunk with overlap
            # Find a good breaking point in the current chunk for overlap
            if len(current_chunk) > overlap:
                # Try to find sentence boundary in the overlap region
                overlap_text = current_chunk[-overlap:]
                boundary = overlap_text.rfind(". ")
                
                if boundary != -1:
                    overlap_start = len(current_chunk) - overlap + boundary + 2
                    current_chunk = current_chunk[overlap_start:]
                else:
                    # If no sentence boundary, just use the last overlap characters
                    current_chunk = current_chunk[-overlap:]
            
            # Add the current sentence to the new chunk
            current_chunk += sentence
        else:
            current_chunk += sentence
    
    # Add the final chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def chunk_by_semantic(text):
    """
    Chunk text based on semantic similarity between sentences.
    
    Args:
        text (str): The text to chunk
        
    Returns:
        list: List of text chunks
    """
    # Check if embeddings service is available
    if not embeddings.test_ollama_connection():
        print("Embeddings service not available. Falling back to size-based chunking.")
        return chunk_by_size(text)
    
    # Get sentences
    sentences = safe_sent_tokenize(text)
    
    # If we have very few sentences, just use size-based chunking
    if len(sentences) < 5:
        logging.info("Too few sentences, falling back to size-based chunking")
        return chunk_by_size(text)
    
    # Generate embeddings for each sentence with progress bar
    logging.info(f"Generating embeddings for {len(sentences)} sentences")
    sentence_embeddings = []
    
    # Use minimal tqdm output 
    for sentence in tqdm(
        sentences, 
        desc="Generating embeddings", 
        unit="sentence",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    ):
        embedding = embeddings.get_embeddings(sentence)
        sentence_embeddings.append(embedding)
    
    # Group sentences into chunks based on semantic similarity
    logging.info("Grouping sentences into chunks based on semantic similarity")
    chunks = []
    current_chunk = sentences[0]
    current_embedding = sentence_embeddings[0]
    
    # Use minimal tqdm output
    for i in tqdm(
        range(1, len(sentences)), 
        desc="Chunking", 
        unit="sentence",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    ):
        sentence = sentences[i]
        embedding = sentence_embeddings[i]
        
        # Calculate similarity between current chunk and next sentence
        similarity = embeddings.cosine_similarity(current_embedding, embedding)
        
        # If similarity is high enough, add to current chunk
        if similarity > SIMILARITY_THRESHOLD and len(current_chunk) + len(sentence) <= MAX_CHUNK_SIZE:
            current_chunk += " " + sentence
            # Update current embedding with a weighted average
            current_len = len(current_chunk) - len(sentence)
            sentence_len = len(sentence)
            weight = current_len / (current_len + sentence_len)
            current_embedding = [weight * a + (1 - weight) * b for a, b in zip(current_embedding, embedding)]
        else:
            # Start a new chunk if the chunk is big enough
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append(current_chunk)
            else:
                # If the chunk is too small, add it to the previous chunk or start a new one
                if chunks:
                    chunks[-1] += " " + current_chunk
                else:
                    chunks.append(current_chunk)
            
            current_chunk = sentence
            current_embedding = embedding
    
    # Add the final chunk
    if current_chunk:
        if len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk)
        elif chunks:
            chunks[-1] += " " + current_chunk
        else:
            chunks.append(current_chunk)
    
    return chunks

def chunk_hybrid(text):
    """
    Hybrid approach that combines structural, semantic, and size-based chunking.
    
    Args:
        text (str): The text to chunk
        
    Returns:
        list: List of text chunks
    """
    # First, split by obvious structural boundaries (headings, etc.)
    logging.info("Splitting text by structural boundaries")
    structural_chunks = split_by_structure(text)
    
    # Process each structural chunk
    logging.info(f"Processing {len(structural_chunks)} structural chunks")
    final_chunks = []
    
    for i, chunk in enumerate(tqdm(
        structural_chunks, 
        desc="Processing chunks", 
        unit="chunk",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )):
        # Skip empty chunks
        if not chunk.strip():
            continue
            
        # If the chunk is small enough, add it directly
        if len(chunk) <= MAX_CHUNK_SIZE:
            final_chunks.append(chunk)
        else:
            # For larger chunks, try semantic chunking if possible
            try:
                semantic_chunks = chunk_by_semantic(chunk)
                final_chunks.extend(semantic_chunks)
            except Exception as e:
                logging.error(f"Error in semantic chunking (chunk {i+1}): {e}")
                # Fall back to size-based chunking
                size_chunks = chunk_by_size(chunk)
                final_chunks.extend(size_chunks)
    
    # Final pass to combine very small chunks
    logging.info("Combining small chunks")
    combined_chunks = []
    current_chunk = ""
    
    for chunk in final_chunks:
        if len(current_chunk) + len(chunk) <= MAX_CHUNK_SIZE:
            if current_chunk:
                current_chunk += "\n\n" + chunk
            else:
                current_chunk = chunk
        else:
            if current_chunk:
                combined_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:
        combined_chunks.append(current_chunk)
    
    return combined_chunks

def split_by_structure(text):
    """
    Split text by structural elements like headers, horizontal rules, etc.
    
    Args:
        text (str): The text to split
        
    Returns:
        list: List of text chunks split by structure
    """
    # Look for heading patterns (Markdown style)
    heading_pattern = r'(?:^|\n)(#{1,6}\s+.+?)(?:\n|$)'
    horizontal_rule = r'(?:^|\n)[-*_]{3,}(?:\n|$)'
    
    # Combine patterns
    split_pattern = f"{heading_pattern}|{horizontal_rule}"
    
    # Find all matches
    matches = list(re.finditer(split_pattern, text))
    
    # If no structure found, return the whole text as one chunk
    if not matches:
        return [text]
    
    chunks = []
    last_end = 0
    
    for match in matches:
        # Add text before this match if it's not empty
        if match.start() > last_end:
            chunk = text[last_end:match.start()].strip()
            if chunk:
                chunks.append(chunk)
        
        # Add the heading/rule itself with the content up to the next match
        chunks.append(text[match.start():match.end()].strip())
        last_end = match.end()
    
    # Add any remaining text after the last match
    if last_end < len(text):
        chunk = text[last_end:].strip()
        if chunk:
            chunks.append(chunk)
    
    return chunks

def process_all_text_files(generate_embeddings=False):
    """
    Process all text files in the data folder.
    Chunks text and optionally generates embeddings for each chunk.
    
    Args:
        generate_embeddings (bool): Whether to generate embeddings for chunks
    """
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_folder = config.get('data_folder', 'data')
    
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' not found.")
        return
    
    print(f"Scanning data folder: {data_folder}...")
    
    # Find all text files that need chunking
    all_text_files = []
    all_folders = []
    
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        all_folders.append(folder_path)
        
        text_path = os.path.join(folder_path, "content.txt")
        if os.path.exists(text_path):
            chunks_dir = os.path.join(folder_path, "chunks")
            
            # Check if chunking is needed (no chunks dir or empty chunks dir)
            if not os.path.exists(chunks_dir) or len(os.listdir(chunks_dir)) == 0:
                all_text_files.append(text_path)
    
    if not all_text_files:
        print("No text files need chunking.")
    else:
        print(f"Found {len(all_text_files)} text files that need chunking.")
        
        # Process each text file and create chunks
        processed_count = 0
        total_chunks = 0
        
        for text_path in tqdm(all_text_files, desc="Chunking files", unit="file"):
            try:
                folder_path = os.path.dirname(text_path)
                chunk_count = chunk_text_file(text_path, method="hybrid")
                processed_count += 1
                total_chunks += chunk_count
            except Exception as e:
                print(f"Error chunking {text_path}: {e}")
                logging.error(f"Error chunking {text_path}: {e}")
    
    # If embeddings generation was requested
    if generate_embeddings:
        print("Scanning for chunks that need embeddings...")
        
        # Find all chunk files for embedding generation
        all_chunk_files = []
        
        for folder_path in all_folders:
            chunks_dir = os.path.join(folder_path, "chunks")
            if os.path.exists(chunks_dir):
                for chunk_file in os.listdir(chunks_dir):
                    if chunk_file.endswith('.txt'):
                        chunk_path = os.path.join(chunks_dir, chunk_file)
                        all_chunk_files.append(chunk_path)
        
        if all_chunk_files:
            print(f"Found {len(all_chunk_files)} chunks total")
            
            try:
                # Import embeddings module
                import embeddings
                
                # Check if Ollama is available
                if not embeddings.test_ollama_connection():
                    print("Warning: Ollama API is not available. Skipping embedding generation.")
                    return
                
                # Filter chunks that don't have embeddings yet
                chunks_needing_embeddings = [path for path in all_chunk_files 
                                           if not embeddings.embedding_exists(path)]
                
                if chunks_needing_embeddings:
                    print(f"Generating embeddings for {len(chunks_needing_embeddings)} chunks without embeddings")
                    embeddings.generate_embeddings_for_chunks(chunks_needing_embeddings)
                else:
                    print("All chunks already have embeddings")
                    
            except Exception as e:
                logging.error(f"Error generating embeddings: {e}")
                print(f"Error generating embeddings: {e}")
        else:
            print("No chunks found for embedding generation")
    
    # Print summary of all chunks
    total_chunks = 0
    for folder_path in all_folders:
        chunks_dir = os.path.join(folder_path, "chunks")
        if os.path.exists(chunks_dir):
            chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.txt')]
            total_chunks += len(chunk_files)
    
    print(f"\nTotal chunks across all folders: {total_chunks}")

if __name__ == "__main__":
    # Test the chunking with a sample text
    process_all_text_files() 