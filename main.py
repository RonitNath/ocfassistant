import ollama
from ollama import chat, ChatResponse, ListResponse, list
from tqdm import tqdm
from embeddings import validate_ollama_setup
import os
import argparse
import json
import time
import logging
from scrape import scraper, is_url_safe
# Import our custom embeddings module
import embeddings
import qdrant
from chunking import process_all_text_files
import yaml

def scrape_data(depth=1, extend_depth=False):
    """
    Test the web scraping functionality with the given depth.
    
    Args:
        depth (int): Maximum crawl depth
        extend_depth (bool): If True, extends an existing crawl to a deeper depth
    """
    print(f"Starting web scraping with depth={depth}{' (extending previous crawl)' if extend_depth else ''}...")
    url_graph = scraper(depth=depth, extend_depth=extend_depth)
    
    # Output some statistics
    print("\nScraping completed. Results:")
    
    # Count the number of scraped pages
    data_folder = "data"
    if os.path.exists(data_folder):
        page_count = 0
        url_count = 0
        text_count = 0
        
        for item in os.listdir(data_folder):
            item_path = os.path.join(data_folder, item)
            if os.path.isdir(item_path):
                page_count += 1
                
                # Count URLs with extracted text
                if os.path.exists(os.path.join(item_path, "content.txt")):
                    text_count += 1
                
                # Count URLs
                if os.path.exists(os.path.join(item_path, "url.txt")):
                    url_count += 1
        
        print(f"- Total pages scraped: {page_count}")
        print(f"- Pages with URL info: {url_count}")
        print(f"- Pages with extracted text: {text_count}")
        
        # Count graph connections
        if url_graph:
            total_links = sum(len(data["links"]) for data in url_graph.values())
            print(f"- Total links in graph: {total_links}")
            print(f"- Average links per page: {total_links/max(1, len(url_graph)):.2f}")
    else:
        print("No data folder found. Scraping may have failed.")

def chat_with_model():
    """
    Chat with the LLM model, using RAG to enhance responses.
    """
    # Setup logging for profiling
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    logging.basicConfig(
        filename='logs/chat_profiling.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Ensure Qdrant collection exists
    qdrant.ensure_collection_exists()
    
    # Get available models
    try:
        models = list()
        print("Available models:")
        for model in models['models']:
            print(f"- {model['name']}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Get model to use from config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config.get('chat_model', 'llama3.2:8b')
    print(f"Using model: {model_name}")
    
    # Validate Ollama setup
    if not validate_ollama_setup():
        print("Ollama setup failed. Please check if Ollama is running.")
        return
    
    print(f"Starting chat with {model_name}. Type 'exit' to quit.")
    
    # Get collection stats
    try:
        stats = qdrant.get_collection_stats()
        print(f"Vector DB contains {stats.get('vectors_count', 0)} documents")
    except Exception as e:
        print(f"Error getting collection stats: {e}")
    
    messages = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting chat.")
            break
        
        # Profile each step of the process
        profile = {}
        
        # Step 1: RAG - Find relevant documents
        start_time = time.time()
        
        # Generate embedding for the query
        try:
            relevant_chunks = qdrant.search_similar_chunks(user_input, limit=10)
            profile['retrieval_time'] = time.time() - start_time
            logging.info(f"Retrieved {len(relevant_chunks)} chunks in {profile['retrieval_time']:.2f} seconds")
            
            # Log the retrieval scores
            if relevant_chunks:
                scores = [chunk['score'] for chunk in relevant_chunks]
                avg_score = sum(scores) / len(scores)
                logging.info(f"Retrieval scores - min: {min(scores):.4f}, max: {max(scores):.4f}, avg: {avg_score:.4f}")
        except Exception as e:
            print(f"Error during retrieval: {e}")
            logging.error(f"Retrieval error: {e}")
            relevant_chunks = []
            profile['retrieval_time'] = time.time() - start_time
        
        # Step 2: Build the prompt with context
        start_time = time.time()
        
        # Format the context from relevant chunks
        context = ""
        if relevant_chunks:
            context = "Context information from the OCF website:\n\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                # Format metadata for citation
                metadata = chunk.get('metadata', {})
                folder_name = metadata.get('folder_name', 'unknown')
                # Get only the text
                text = chunk.get('original_text', '')
                # Add the chunk to the context with a citation
                context += f"[{i}] From {folder_name}:\n{text}\n\n"
        
        profile['context_preparation_time'] = time.time() - start_time
        logging.info(f"Context preparation took {profile['context_preparation_time']:.2f} seconds")
        
        # Step 3: Send the query to the model with context
        start_time = time.time()
        
        # Add the user message
        messages.append({"role": "user", "content": user_input})
        
        # If we have context, add system message
        if context:
            # Add a system message with the context
            system_message = {
                "role": "system",
                "content": f"You are an assistant for the Open Computing Facility at UC Berkeley. "
                           f"Use the following information to answer the user's question. "
                           f"If the information provided doesn't answer the question, say so and provide "
                           f"general information about OCF if available.\n\n{context}"
            }
            
            # Insert the context at the beginning of the conversation
            if len(messages) == 1:  # Only the current user message
                messages = [system_message] + messages
            else:
                # Replace the existing system message if any
                system_msg_index = next((i for i, msg in enumerate(messages) if msg["role"] == "system"), None)
                if system_msg_index is not None:
                    messages[system_msg_index] = system_message
                else:
                    messages = [system_message] + messages
        
        try:
            # Call the model
            response = chat(model=model_name, messages=messages)
            profile['model_response_time'] = time.time() - start_time
            logging.info(f"Model response took {profile['model_response_time']:.2f} seconds")
            
            # Add the assistant's response to the conversation history
            messages.append({"role": "assistant", "content": response["message"]["content"]})
            
            # Calculate total processing time
            total_time = profile.get('retrieval_time', 0) + profile.get('context_preparation_time', 0) + profile.get('model_response_time', 0)
            
            print(f"\nAssistant: {response['message']['content']}")
            print(f"\n[Processing took {total_time:.2f}s: Retrieval {profile.get('retrieval_time', 0):.2f}s, "
                  f"Context {profile.get('context_preparation_time', 0):.2f}s, "
                  f"Model {profile.get('model_response_time', 0):.2f}s]")
            
            # Log overall performance
            logging.info(f"Total processing time: {total_time:.2f} seconds")
            logging.info(f"Profile: {profile}")
        
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Model response error: {e}")

def chunk_text():
    """
    Run semantic chunking of text data without generating embeddings.
    """
    print("Starting semantic chunking of text data...")
    try:
        from chunking import process_all_text_files
        # Process all text files but don't generate embeddings
        process_all_text_files(generate_embeddings=False)
        print("Chunking completed successfully.")
        
        # Count the number of chunks created
        data_folder = "data"
        if os.path.exists(data_folder):
            total_chunks = 0
            processed_files = 0
            
            for folder_name in os.listdir(data_folder):
                folder_path = os.path.join(data_folder, folder_name)
                
                # Skip if not a directory
                if not os.path.isdir(folder_path) or folder_name == "url_graph.json":
                    continue
                    
                chunks_dir = os.path.join(folder_path, "chunks")
                if os.path.exists(chunks_dir):
                    processed_files += 1
                    # Count chunk files
                    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.txt')]
                    total_chunks += len(chunk_files)
                    
            print(f"- Total files processed: {processed_files}")
            print(f"- Total chunks created: {total_chunks}")
            if processed_files > 0:
                print(f"- Average chunks per file: {total_chunks/processed_files:.2f}")
    except Exception as e:
        print(f"Error during chunking: {e}")
        logging.error(f"Error during chunking: {e}")

def simulate_scrape(target_depth, debug=False, estimate_unreachable=False):
    """
    Simulate scraping to a target depth without fetching any URLs.
    Uses the existing URL graph to predict how many new URLs would be fetched.
    
    Args:
        target_depth (int): The maximum depth to simulate
        debug (bool): Whether to show detailed debug information
        estimate_unreachable (bool): If True, include unreachable URLs in the estimation
        
    Returns:
        dict: Statistics about the simulated crawl
    """
    data_folder = "data"
    graph_path = os.path.join(data_folder, "url_graph.json")
    
    if not os.path.exists(graph_path):
        print("No URL graph found. Run scraper first to create initial graph.")
        return None
    
    # Load the existing URL graph
    print(f"Loading existing URL graph to simulate crawl to depth {target_depth}...")
    with open(graph_path, 'r', encoding='utf-8') as f:
        url_graph = json.load(f)
    
    # Find current maximum depth in the graph
    current_max_depth = 0
    if url_graph:
        depths = [info.get('depth', 0) for info in url_graph.values()]
        current_max_depth = max(depths) if depths else 0
    
    print(f"Current maximum depth in graph: {current_max_depth}")
    
    if target_depth <= current_max_depth:
        print(f"Target depth {target_depth} is not greater than current depth {current_max_depth}")
        print("To simulate extending depth, use a target_depth greater than the current maximum.")
        return None
    
    # Maps URLs to their depth
    url_to_depth = {}
    for _, info in url_graph.items():
        url = info.get('url')
        if url:
            # Some URLs might not have depth info, assume they're at depth 0
            depth = info.get('depth', 0)
            url_to_depth[url] = depth
    
    # Maps URLs to their outgoing links
    url_to_links = {}
    
    # First, create a mapping from sanitized URL to actual URL for quick lookup
    sanitized_to_url = {}
    
    # Build a mapping from sanitized domain names to full URLs
    for content_hash, info in url_graph.items():
        url = info.get('url', '')
        if not url:
            continue
            
        # Extract the domain and path for matching with link entries
        if '://' in url:
            try:
                domain = url.split('://')[1].replace('www.', '')
                domain = domain.rstrip('/')  # Remove trailing slash
                
                # Store the domain mapping
                sanitized_domain = domain.replace('.', '_').replace('/', '.')
                sanitized_to_url[sanitized_domain] = url
                
                # Also store simpler versions for matching
                parts = domain.split('/')
                base_domain = parts[0]
                sanitized_base = base_domain.replace('.', '_')
                sanitized_to_url[sanitized_base] = url
                
                # Store with dots instead of underscores (both formats appear in the data)
                sanitized_to_url[domain.replace('/', '.')] = url
            except IndexError:
                continue
    
    # Maps URLs to their outgoing links, resolving the sanitized domains to full URLs
    for content_hash, info in url_graph.items():
        url = info.get('url', '')
        if not url:
            continue
            
        links = []
        for link_name in info.get('links', []):
            # Try to find the full URL that corresponds to this sanitized link name
            if link_name in sanitized_to_url:
                target_url = sanitized_to_url[link_name]
                links.append(target_url)
                
        url_to_links[url] = links
    
    # Track URLs at each depth
    urls_by_depth = {d: set() for d in range(current_max_depth + 1, target_depth + 1)}
    
    # Get URLs at current max depth to start simulation from
    current_depth_urls = set()
    for url, depth in url_to_depth.items():
        if depth == current_max_depth:
            # Only include URLs that have outgoing links
            if url in url_to_links and url_to_links[url]:
                current_depth_urls.add(url)
    
    print(f"Found {len(current_depth_urls)} URLs at depth {current_max_depth} with outgoing links")
    
    # If we don't have any URLs at the max depth, try depth-1 as well
    if not current_depth_urls and current_max_depth > 0:
        for url, depth in url_to_depth.items():
            if depth == current_max_depth - 1:
                if url in url_to_links and url_to_links[url]:
                    current_depth_urls.add(url)
        print(f"Used fallback: Found {len(current_depth_urls)} URLs at depth {current_max_depth-1} with outgoing links")
    
    # Simulate the crawl
    for depth in range(current_max_depth + 1, target_depth + 1):
        print(f"Simulating crawl at depth {depth}...")
        next_depth_urls = set()
        
        # For each URL at the previous depth
        for url in current_depth_urls:
            # Get its outgoing links
            outgoing_links = url_to_links.get(url, [])
            
            # Add new links to the current depth
            for link in outgoing_links:
                # Don't include URLs we've already processed at a lower or equal depth
                if link not in url_to_depth or url_to_depth[link] > depth:
                    # Check if URL is safe (would be fetched by real scraper)
                    if is_url_safe(link):
                        urls_by_depth[depth].add(link)
                        # This URL will be considered for the next depth
                        url_to_depth[link] = depth
                        next_depth_urls.add(link)
        
        # Update for next iteration
        current_depth_urls = next_depth_urls
        print(f"  - Found {len(urls_by_depth[depth])} new URLs at depth {depth}")
        print(f"  - {len(next_depth_urls)} URLs with outgoing links will be used for depth {depth+1}")
    
    # Create a set of unresolved links to consider as potential new URLs
    unresolved_links = set()
    if estimate_unreachable:
        for _, info in url_graph.items():
            link_names = info.get('links', [])
            for link_name in link_names:
                if link_name not in sanitized_to_url:
                    unresolved_links.add(link_name)
        
        print(f"Found {len(unresolved_links)} unique unresolved links that could be additional URLs")
        
        # Add these as potential new URLs at the next depth
        if unresolved_links and current_max_depth < target_depth:
            urls_by_depth[current_max_depth + 1].update(unresolved_links)
            print(f"Added {len(unresolved_links)} potential new URLs at depth {current_max_depth + 1}")
    
    # Calculate statistics
    total_new_urls = sum(len(urls) for depth, urls in urls_by_depth.items())
    urls_per_depth = {depth: len(urls) for depth, urls in urls_by_depth.items()}
    
    # Print results
    print("\nSimulation Results:")
    print(f"- Total URLs in current graph: {len(url_graph)}")
    print(f"- Estimated new URLs to be fetched: {total_new_urls}")
    
    print("\nBreakdown by depth:")
    for depth in range(current_max_depth + 1, target_depth + 1):
        print(f"  Depth {depth}: {urls_per_depth.get(depth, 0)} new URLs")
    
    # Count how many links were resolved
    total_links = sum(len(info.get('links', [])) for info in url_graph.values())
    resolved_links = sum(len(links) for links in url_to_links.values())
    print(f"Resolved {resolved_links} out of {total_links} links ({resolved_links/max(1,total_links)*100:.1f}%)")
    
    # Debug information if requested
    if debug:
        print("\nSample of resolved links:")
        sample_count = 0
        for url, links in url_to_links.items():
            if links and sample_count < 5:
                print(f"URL: {url}")
                print(f"  Resolved links: {links[:5]}")
                sample_count += 1
        
        # Check for unrealized links
        print("\nAnalyzing a sample of unresolved links:")
        sample_count = 0
        for _, info in url_graph.items():
            if info.get('links') and sample_count < 5:
                url = info.get('url')
                link_names = info.get('links')
                resolved = [link_name for link_name in link_names if link_name in sanitized_to_url]
                unresolved = [link_name for link_name in link_names if link_name not in sanitized_to_url]
                
                if unresolved:
                    print(f"URL: {url}")
                    print(f"  Unresolved links ({len(unresolved)}): {unresolved[:5]}")
                    sample_count += 1
    
    return {
        "current_urls": len(url_graph),
        "new_urls": total_new_urls,
        "urls_by_depth": urls_per_depth
    }

def upload_to_qdrant():
    """
    Upload all embeddings to the Qdrant vector database.
    """
    print("Starting upload of embeddings to Qdrant...")
    
    # Ensure Qdrant collection exists
    if not qdrant.ensure_collection_exists():
        print("Failed to ensure Qdrant collection exists.")
        return
    
    # Process all embeddings
    success_count, total_count = qdrant.process_all_embeddings()
    
    print(f"\nQdrant upload completed:")
    print(f"- Successfully uploaded: {success_count}/{total_count} embeddings")
    
    # Get collection stats
    stats = qdrant.get_collection_stats()
    print(f"- Collection now contains {stats.get('vectors_count', 0)} vectors")

def generate_embeddings():
    """
    Generate embeddings for all chunks that don't already have them.
    """
    print("Generating embeddings for chunks...")
    try:
        from chunking import process_all_text_files
        # Process text files with embedding generation enabled
        process_all_text_files(generate_embeddings=True)
        print("Successfully completed embedding generation process.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        logging.error(f"Error generating embeddings: {e}")

def web_server():
    """Start a web server to serve the OCF Assistant frontend."""
    try:
        import flask
        from flask import Flask, send_from_directory, render_template
        from flask_socketio import SocketIO, emit
        import os
        import time
        import threading
        
        app = Flask(__name__, 
                   static_folder="static",
                   template_folder="templates")
        socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=30, ping_interval=15)
        
        # Track connected clients
        connected_clients = set()
        
        @app.route('/')
        def index():
            return render_template('index.html')
            
        @app.route('/<path:path>')
        def serve_static(path):
            return send_from_directory('static', path)
        
        @socketio.on('connect')
        def handle_connect():
            client_id = flask.request.sid
            connected_clients.add(client_id)
            print(f"Client connected: {client_id}")
            emit('connection_response', {'status': 'connected', 'client_id': client_id})
        
        @socketio.on('disconnect')
        def handle_disconnect():
            client_id = flask.request.sid
            if client_id in connected_clients:
                connected_clients.remove(client_id)
            print(f"Client disconnected: {client_id}")
        
        @socketio.on('ping')
        def handle_ping(data):
            client_id = flask.request.sid
            print(f"Ping from client {client_id}: {data}")
            emit('pong', {'timestamp': time.time(), 'received': data})
        
        def background_tasks():
            """Background task to send status updates to all connected clients"""
            while True:
                socketio.sleep(5)  # Update every 5 seconds
                if connected_clients:
                    status_data = {
                        'timestamp': time.time(),
                        'connected_clients': len(connected_clients)
                    }
                    socketio.emit('status_update', status_data)
                    
        # Start background thread for periodic updates
        socketio.start_background_task(background_tasks)
            
        print("Starting web server with WebSocket support on http://localhost:8080")
        socketio.run(app, host='0.0.0.0', port=8080, debug=True, allow_unsafe_werkzeug=True)
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Flask and Flask-SocketIO are required to run the web interface.")
        print("Install them with 'pip install flask flask-socketio'.")
    except Exception as e:
        print(f"Error starting web server: {e}")
        logging.error(f"Error starting web server: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCF Assistant CLI")
    parser.add_argument("--scrape", action="store_true", help="Run web scraping")
    parser.add_argument("--depth", type=int, default=1, help="Depth of scraping (default: 1)")
    parser.add_argument("--extend-depth", action="store_true", help="Extend previous crawl to deeper depth")
    parser.add_argument("--chunk", action="store_true", help="Run chunking on scraped text")
    parser.add_argument("--chat", action="store_true", help="Chat with the model")
    parser.add_argument("--simulate", type=int, help="Simulate scraping to specified depth without fetching URLs")
    parser.add_argument("--debug", action="store_true", help="Show detailed debug information")
    parser.add_argument("--estimate-unreachable", action="store_true", help="Include unreachable URLs in simulation estimates")
    parser.add_argument("--qdrant", action="store_true", help="Upload embeddings to Qdrant vector database")
    parser.add_argument("--embeddings", action="store_true", help="Generate embeddings for chunks")
    parser.add_argument("--web", action="store_true", help="Start web interface on port 8080")
    
    args = parser.parse_args()
    
    from utils import setup_env
    _ = setup_env() # For ollama and qdrant urls
    
    if args.scrape:
        scrape_data(depth=args.depth, extend_depth=args.extend_depth)
    elif args.chat:
        chat_with_model()
    elif args.chunk:
        chunk_text()
    elif args.simulate is not None:
        simulate_scrape(args.simulate, debug=args.debug, estimate_unreachable=args.estimate_unreachable)
    elif args.qdrant:
        upload_to_qdrant()
    elif args.embeddings:
        generate_embeddings()
    elif args.web:
        web_server()
    else:
        parser.print_help()
