import os
import yaml
import json
import logging
import time
from tqdm import tqdm
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import embeddings

# Load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure logging
logging.basicConfig(
    filename='logs/qdrant.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get Qdrant URL from environment variable
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
collection_name = config.get('qdrant_collection', 'ocf_collection')
embedding_dim = config.get('embedding_dimension', 4096)  # Default for many Ollama models

# Initialize client
client = QdrantClient(url=qdrant_url)

def ensure_collection_exists():
    """
    Ensure the collection exists, creating it if necessary.
    """
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            print(f"Creating collection '{collection_name}'...")
            
            # Create the collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,  # Index every point
                ),
            )
            
            # Create payload index to speed up filtering
            client.create_payload_index(
                collection_name=collection_name,
                field_name="folder_name",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            
            client.create_payload_index(
                collection_name=collection_name,
                field_name="chunk_index",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
            
            print(f"Collection '{collection_name}' created successfully")
            return True
        
        return True
    except Exception as e:
        logging.error(f"Error ensuring collection exists: {e}")
        print(f"Error: {e}")
        return False

def get_point_id_for_chunk(chunk_path):
    """
    Generate a deterministic point ID for a chunk.
    
    Args:
        chunk_path (str): Path to the chunk file
        
    Returns:
        str: A unique ID for the point
    """
    # Use the chunk path to create a deterministic ID
    return str(abs(hash(chunk_path)) % (10 ** 10))

def get_original_text_from_chunk(chunk_path):
    """
    Get the original text from a chunk file.
    
    Args:
        chunk_path (str): Path to the chunk file
        
    Returns:
        str: The text content of the chunk
    """
    try:
        with open(chunk_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading chunk {chunk_path}: {e}")
        return ""

def is_chunk_in_collection(chunk_path):
    """
    Check if a chunk is already in the collection.
    
    Args:
        chunk_path (str): Path to the chunk file
        
    Returns:
        bool: True if the chunk is in the collection, False otherwise
    """
    try:
        point_id = get_point_id_for_chunk(chunk_path)
        
        # Check if the point exists
        response = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
        )
        
        return len(response) > 0
    except Exception as e:
        logging.error(f"Error checking if chunk is in collection: {e}")
        return False

def upload_chunk_embedding(chunk_path, batch_mode=False):
    """
    Upload a chunk embedding to the vector database.
    
    Args:
        chunk_path (str): Path to the chunk file
        batch_mode (bool): Whether this is being called in batch mode
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not ensure_collection_exists():
        return False
    
    # Check if chunk already exists in the collection
    if not batch_mode and is_chunk_in_collection(chunk_path):
        logging.info(f"Chunk {chunk_path} is already in the collection")
        return True
    
    try:
        # Load embedding and metadata
        if not embeddings.embedding_exists(chunk_path):
            logging.error(f"No embedding found for {chunk_path}")
            return False
        
        vector, metadata = embeddings.load_embedding(chunk_path)
        
        # Get the original text
        original_text = get_original_text_from_chunk(chunk_path)
        
        # Prepare the point
        point_id = get_point_id_for_chunk(chunk_path)
        
        # Prepare metadata for storage
        payload = metadata.copy()
        payload['original_text'] = original_text
        payload['chunk_path'] = chunk_path
        
        # Add the point to the collection
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                )
            ]
        )
        
        logging.info(f"Successfully uploaded chunk {chunk_path} to collection")
        return True
        
    except Exception as e:
        logging.error(f"Error uploading chunk {chunk_path}: {e}")
        return False

def upload_embeddings_batch(chunk_paths):
    """
    Upload multiple chunk embeddings to the vector database in batch.
    
    Args:
        chunk_paths (list): List of paths to chunk files
        
    Returns:
        tuple: (success_count, total_count)
    """
    if not ensure_collection_exists():
        return 0, len(chunk_paths)
    
    # Get existing points to avoid re-uploading
    all_point_ids = [get_point_id_for_chunk(path) for path in chunk_paths]
    
    try:
        existing_points = set()
        
        # Check which points already exist in batches
        batch_size = 100
        for i in range(0, len(all_point_ids), batch_size):
            batch_ids = all_point_ids[i:i+batch_size]
            response = client.retrieve(
                collection_name=collection_name,
                ids=batch_ids,
            )
            
            existing_points.update([point.id for point in response])
        
        # Filter out chunks that already exist
        chunks_to_upload = [
            path for path, point_id in zip(chunk_paths, all_point_ids)
            if point_id not in existing_points and embeddings.embedding_exists(path)
        ]
        
        if not chunks_to_upload:
            print("All chunks are already in the collection")
            return 0, len(chunk_paths)
        
        print(f"Uploading {len(chunks_to_upload)} new chunks to Qdrant...")
        
        # Prepare batch of points
        points = []
        
        for chunk_path in tqdm(chunks_to_upload, desc="Preparing points", unit="chunk"):
            try:
                # Load embedding and metadata
                vector, metadata = embeddings.load_embedding(chunk_path)
                
                # Get the original text
                original_text = get_original_text_from_chunk(chunk_path)
                
                # Prepare metadata for storage
                payload = metadata.copy()
                payload['original_text'] = original_text
                payload['chunk_path'] = chunk_path
                
                # Add to batch
                points.append(
                    models.PointStruct(
                        id=get_point_id_for_chunk(chunk_path),
                        vector=vector.tolist(),
                        payload=payload
                    )
                )
                
            except Exception as e:
                logging.error(f"Error preparing point for {chunk_path}: {e}")
        
        # Upload in smaller batches to avoid timeouts
        batch_size = 100
        success_count = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                success_count += len(batch)
            except Exception as e:
                logging.error(f"Error uploading batch {i//batch_size + 1}: {e}")
        
        print(f"Successfully uploaded {success_count} out of {len(chunks_to_upload)} chunks")
        return success_count, len(chunk_paths)
        
    except Exception as e:
        logging.error(f"Error in batch upload: {e}")
        return 0, len(chunk_paths)

def search_similar_chunks(query_text, limit=10, filter_folders=None):
    """
    Search for similar chunks to a query text.
    
    Args:
        query_text (str): Query text to search for
        limit (int): Maximum number of results to return
        filter_folders (list): List of folder names to filter by
        
    Returns:
        list: List of search results with metadata
    """
    if not ensure_collection_exists():
        return []
    
    try:
        # Generate embedding for the query
        start_time = time.time()
        query_vector = embeddings.get_embeddings(query_text)
        embedding_time = time.time() - start_time
        logging.info(f"Generated query embedding in {embedding_time:.2f} seconds")
        
        # Prepare filter if needed
        filter_query = None
        if filter_folders:
            filter_query = models.Filter(
                must=[
                    models.FieldCondition(
                        key="folder_name",
                        match=models.MatchAny(any=filter_folders)
                    )
                ]
            )
        
        # Search for similar chunks
        start_time = time.time()
        response = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=limit,
            filter=filter_query,
            with_payload=True,
        )
        search_time = time.time() - start_time
        logging.info(f"Performed vector search in {search_time:.2f} seconds")
        
        # Format results
        results = []
        for point in response:
            result = {
                'id': point.id,
                'score': point.score,
                'original_text': point.payload.get('original_text', ''),
                'metadata': {k: v for k, v in point.payload.items() if k != 'original_text'},
                'chunk_path': point.payload.get('chunk_path', '')
            }
            results.append(result)
        
        return results
    
    except Exception as e:
        logging.error(f"Error searching for similar chunks: {e}")
        return []

def process_all_embeddings():
    """
    Process all embeddings and upload them to Qdrant.
    
    Returns:
        tuple: (success_count, total_count)
    """
    data_folder = config.get('data_folder', 'data')
    
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' not found.")
        return 0, 0
    
    print(f"Scanning data folder: {data_folder} for embeddings...")
    
    # Find all embeddings
    all_chunk_files = []
    
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        chunks_dir = os.path.join(folder_path, "chunks")
        if os.path.exists(chunks_dir):
            for chunk_file in os.listdir(chunks_dir):
                if chunk_file.endswith('.txt'):
                    chunk_path = os.path.join(chunks_dir, chunk_file)
                    if embeddings.embedding_exists(chunk_path):
                        all_chunk_files.append(chunk_path)
    
    if not all_chunk_files:
        print("No embeddings found to upload.")
        return 0, 0
    
    print(f"Found {len(all_chunk_files)} chunks with embeddings to process")
    
    # Upload embeddings in batch
    return upload_embeddings_batch(all_chunk_files)

def get_collection_stats():
    """
    Get statistics about the collection.
    
    Returns:
        dict: Collection statistics
    """
    try:
        if not ensure_collection_exists():
            return {"error": "Collection does not exist"}
        
        collection_info = client.get_collection(collection_name)
        
        stats = {
            "collection_name": collection_name,
            "vectors_count": collection_info.vectors_count,
            "status": str(collection_info.status),
            "dimension": embedding_dim,
        }
        
        return stats
    except Exception as e:
        logging.error(f"Error getting collection stats: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test the Qdrant integration
    ensure_collection_exists()
    print("Qdrant collection checked/created.")
    
    # Process embeddings
    success_count, total_count = process_all_embeddings()
    print(f"Processed {success_count} out of {total_count} embeddings")
    
    # Print stats
    stats = get_collection_stats()
    print(f"Collection stats: {stats}")

