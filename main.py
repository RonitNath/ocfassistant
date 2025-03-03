import ollama
from ollama import chat, ChatResponse, ListResponse, list
from tqdm import tqdm
from embeddings import validate_ollama_setup
import os
import argparse
from scrape import scraper
# Import our custom embeddings module
import embeddings
from chunking import process_all_text_files

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
    """Chat with the LLM model."""
    model_name = 'llama3.2:3b'
    print("Currently, write your own prompt in config.yml's example_question")
    print("Using model:", model_name)
    
    # Ensure the model is available
    validate_ollama_setup(model_name)
    
    import asyncio
    from chat import run_chat

    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print('\nGoodbye!')

def chunk_text():
    """Test the semantic chunking functionality."""
    print("Starting semantic chunking of text data...")
    process_all_text_files()
    
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
        
        print(f"\nChunking completed. Results:")
        print(f"- Files processed: {processed_files}")
        print(f"- Total chunks created: {total_chunks}")
        print(f"- Average chunks per file: {total_chunks/max(1, processed_files):.2f}")
    else:
        print("No data folder found. Chunking may have failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCF Assistant tools")
    parser.add_argument("--scrape", action="store_true", help="Run the web scraper")
    parser.add_argument("--depth", type=int, default=1, help="Web scraping depth (default: 1)")
    parser.add_argument("--extend-depth", action="store_true", help="Extend existing web scrape to a greater depth")
    parser.add_argument("--chat", action="store_true", help="Chat with the LLM model")
    parser.add_argument("--chunk", action="store_true", help="Run semantic chunking on scraped data")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Model to use for chat and chunking (default: llama3.2:3b)")

    args = parser.parse_args()

    from utils import setup_env
    _ = setup_env() # For ollama and qdrant urls
    
    if args.scrape:
        scrape_data(depth=args.depth, extend_depth=args.extend_depth)
    elif args.chat:
        chat_with_model()
    elif args.chunk:
        chunk_text()
    else:
        # Default behavior if no arguments provided
        print("No action specified. Use --scrape to run web scraper, --chunk to run semantic chunking, or --chat to chat with the model.")
        print("Run 'python main.py --help' for more information.")
