### Obtains preliminary data from the web. Should only need to be run once. (Check if the data is already scraped)

import requests
from bs4 import BeautifulSoup
import json
import hashlib
import os
import re
from urllib.parse import urljoin, urlparse
import time
from fnmatch import fnmatch
from tqdm import tqdm
import datetime
import time
import logging

# Get the config
with open('config.yml', 'r') as f:
    from yaml import safe_load

    config = safe_load(f)

data_folder = config['data_folder']
scrape_safe_patterns = config['scrape_safe_patterns']
scrape_root = config['scrape_root']

def scraper(depth=1, extend_depth=False):
    """
    Use scrape_root to get the intiial document
    Get all the links from the initial document, and filter them using scrape_safe_patterns

    For each page of data, compute the hash 6-bit hash of the page
    If the hash is not in the data_folder, save the page to the data_folder, titled "safe-url-<hash>.html"
    If the hash is in the data_folder, skip the page

    Create a graph of the links between the data in json format, with the following schema:
    {
        "url": "<sanitized-url-hash>",
        "links": ["<sanitized-url-hash>", "<sanitized-url-hash>", ...],
        "depth": depth_at_which_url_was_found,
        "leaf": true_if_no_outgoing_links
    }

    Links are always extracted and stored for all pages, even if beyond the current depth.
    This allows for re-running with increased depth without needing to re-fetch already visited pages.
    Only links within the current depth limit are followed during crawling.

    Args:
        depth (int): Maximum crawl depth
        extend_depth (bool): If True, extends an existing crawl to a deeper depth

    Then go through each file in the data_folder, and for each folder which does not have a .txt file,
    read the html file and get the text from the file.

    Then go through each file in the data_folder, and for each folder which does not have a .embeddings.txt file,
    compute the embeddings of the text using ollama.

    Then save the embeddings to the .embeddings.txt file.
    """
    print("Starting web scraping process...")
    
    # Ensure data directory exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Dictionary to store the URL graph
    url_graph = {}
    
    # If we're extending an existing graph, load it
    graph_path = os.path.join(data_folder, "url_graph.json")
    previous_depth = 0
    if extend_depth and os.path.exists(graph_path):
        print("Loading existing URL graph to extend depth...")
        with open(graph_path, 'r', encoding='utf-8') as f:
            url_graph = json.load(f)
        
        # Find the maximum depth in the current graph
        if url_graph:
            depths = [info.get('depth', 0) for info in url_graph.values()]
            previous_depth = max(depths) if depths else 0
            print(f"Found existing graph with maximum depth {previous_depth}")
            
            if depth <= previous_depth:
                print(f"Warning: New depth {depth} is not greater than previous depth {previous_depth}")
                print("To re-crawl at the same depth, don't use extend_depth=True")
    
    visited_urls = set()
    
    # If extending depth, get URLs from the frontier of the previous crawl
    if extend_depth and previous_depth > 0:
        urls_to_visit = get_urls_for_depth_extension(previous_depth, depth)
        # If we found URLs to extend, we should mark all previously visited URLs
        for content_hash, info in url_graph.items():
            visited_urls.add(info.get('url', ''))
    else:
        # Start fresh from root URLs
        urls_to_visit = [(url, 0) for url in scrape_root]  # (url, depth)
    
    # Dictionary to map URLs to their folder names
    url_to_folder = {}
    
    # For the progress bar, we'll use an estimated number of URLs to visit
    estimated_urls = len(urls_to_visit) * (depth + 1)  # Simple estimation
    

    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Clear the log file
    with open("logs/scraping_profiling.txt", "w") as f:
        f.truncate(0)

    # Set up logging for profiling
    logging.basicConfig(filename='logs/scraping_profiling.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Stage 1: URL crawling with progress bar and profiling
    print(f"Stage 1: Crawling URLs")
    pbar = tqdm(total=estimated_urls, desc="Crawling", unit="URL")
    
    processed_count = 0
    start_time = time.time()  # Start time for the entire crawling process
    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)
        
        # Update progress bar
        pbar.update(1)
        processed_count += 1
        
        # Skip if already visited
        if current_url in visited_urls:
            continue
            
        # Check if URL matches safe patterns
        if not is_url_safe(current_url):
            continue
            
        # Update progress bar description with current URL (shortened)
        short_url = current_url[:40] + "..." if len(current_url) > 40 else current_url
        pbar.set_description(f"Crawling: {short_url}")
        
        visited_urls.add(current_url)
        
        # Get content and process the page
        try:
            content = fetch_url(current_url)
            logging.info(f"Fetching finished for {current_url} in {time.time() - start_time} seconds")
            if not content:
                continue
                
            url_hash = hash_content(content)
            logging.info(f"Hashing finished for {current_url} in {time.time() - start_time} seconds")

            sanitized_url = sanitize_url(current_url)
            folder_name, is_duplicate, duplicate_of = save_html_content(url_hash, current_url, content)
            
            # Store mapping of URL to folder name
            url_to_folder[current_url] = folder_name
            
            # Always extract links regardless of depth
            links = extract_links(current_url, content)
            logging.info(f"Extracting links finished for {current_url} in {time.time() - start_time} seconds")
            link_folders = []
            
            # If we found new links, update our estimated total
            new_links = [link for link in links if link not in visited_urls and is_url_safe(link)]
            if new_links and current_depth < depth:
                pbar.total += len(new_links)
                pbar.refresh()
            
            for link in links:
                if is_url_safe(link):  # Always store safe links in the graph
                    link_sanitized = sanitize_url(link)
                    link_folder = f"{link_sanitized}"
                    link_folders.append(link_folder)
                    
                    # Only add to queue if we're within the depth limit
                    if link not in visited_urls and current_depth < depth:
                        urls_to_visit.append((link, current_depth + 1))
            
            logging.info(f"Adding to queue finished for {current_url} in {time.time() - start_time} seconds")

            # Add to graph with depth and leaf info
            content_hash = folder_name.split('-')[-1]  # Extract the hash part from the folder name
            url_graph[content_hash] = {
                "url": current_url,
                "links": link_folders,
                "depth": current_depth,
                "leaf": len(link_folders) == 0  # True if no outgoing links
            }
            
            # Log the time taken for this iteration
            iteration_end_time = time.time()
            logging.info(f"Processed {current_url} in {iteration_end_time - start_time} seconds")
            
        except Exception as e:
            pbar.write(f"Error processing {current_url}: {e}")
            # Log the error
            logging.error(f"Error processing {current_url}: {e}")
            # Check if URL is already in failed_urls.txt before adding
            failed_urls_path = "logs/failed_urls.txt"
            
            # First check if the URL is already in the file
            if os.path.exists(failed_urls_path):
                with open(failed_urls_path, "r") as f:
                    existing_urls = f.read()
            else:
                existing_urls = ""
                
            # Then append if not already in the file
            if current_url not in existing_urls:
                with open(failed_urls_path, "a") as f:
                    f.write(current_url + "\n")
    
    pbar.close()
    print(f"Completed crawling {processed_count} URLs.")
    
    # Stage 2: Save the URL graph
    print("Stage 2: Saving URL graph...")
    save_url_graph(url_graph)
    
    # Stage 3: Process HTML files to extract text
    print("Stage 3: Extracting text from HTML files...")
    process_html_to_text()
    
    return url_graph

def is_url_safe(url):
    """Check if a URL matches any of the safe patterns."""
    return any(fnmatch(url, pattern) for pattern in scrape_safe_patterns)

def hash_content(content):
    """Create a 6-character hash of HTML content."""
    return hashlib.sha256(content.encode()).hexdigest()[:6]

def fetch_url(url):
    """Fetch the content of a URL."""
    try:
        timeout = config['scrape_fetch_timeout']
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f" Failed to fetch {url}: {e}")
        # Check if failed url is already in failed_urls.txt before adding
        failed_urls_path = "logs/failed_urls.txt"
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(failed_urls_path), exist_ok=True)
        
        # First check if the URL is already in the file
        if os.path.exists(failed_urls_path):
            with open(failed_urls_path, "r") as f:
                existing_urls = f.read()
        else:
            existing_urls = ""
            
        # Then append if not already in the file
        if url not in existing_urls:
            with open(failed_urls_path, "a") as f:
                f.write(url + "\n")
        return None

def extract_links(base_url, html_content):
    """Extract and normalize all links from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        
        # Remove fragments and queries
        parsed = urlparse(full_url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        if clean_url.startswith(('http://', 'https://')):
            links.append(clean_url)
    
    return links

def save_html_content(content_hash, original_url, content):
    """
    Save HTML content to a file, handling duplicate content.
    
    Returns:
        tuple: (folder_name, is_duplicate, duplicate_of)
    """
    # Track if this is a duplicate and what it's a duplicate of
    is_duplicate = False
    duplicate_of = None
    
    # Create a sanitized version of the URL for the folder name
    sanitized_url = sanitize_url(original_url)
    folder_name = f"{sanitized_url}-{content_hash}"
    
    # Check if this content hash already exists in a different folder
    for existing_dir in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, existing_dir)) and existing_dir.endswith(f"-{content_hash}"):
            # Found a duplicate!
            if existing_dir != folder_name:  # Not the same URL
                is_duplicate = True
                duplicate_of = existing_dir
                break
    
    # Create directory for the folder if it doesn't exist
    hash_dir = os.path.join(data_folder, folder_name)
    if not os.path.exists(hash_dir):
        os.makedirs(hash_dir)
    
    # Create metadata.json with URL information
    metadata = {
        "url": original_url,
        "hash": content_hash,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    
    # If this is a duplicate, add that information to metadata
    if is_duplicate:
        metadata["is_duplicate"] = True
        metadata["duplicate_of"] = duplicate_of
    
    # Save metadata
    with open(os.path.join(hash_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Save the HTML content (only if not a duplicate)
    if not is_duplicate:
        html_path = os.path.join(hash_dir, "content.html")
        if not os.path.exists(html_path):
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    return folder_name, is_duplicate, duplicate_of

def sanitize_url(url):
    """
    Convert a URL to a sanitized string suitable for a folder name.
    Example: "https://ocf.io/docs/services.html" -> "ocfio.docs.services"
    """
    # Remove protocol (http://, https://)
    url = re.sub(r'^https?://', '', url)
    
    # Remove trailing slashes
    url = url.rstrip('/')
    
    # Replace special characters
    url = url.replace('www.', '')
    
    # Remove file extensions
    url = re.sub(r'\.(html|php|asp|jsp)$', '', url)
    
    # Replace slashes, dots, and other characters with appropriate delimiters
    url = url.replace('/', '.').replace('-', '_')
    
    # Remove any remaining invalid characters for folder names
    url = re.sub(r'[^a-zA-Z0-9_.]', '', url)
    
    # Limit the length to avoid extremely long folder names
    if len(url) > 50:
        url = url[:50]
    
    return url

def save_url_graph(url_graph):
    """
    Save the URL graph to a JSON file.
    
    The URL graph now uses content hashes as keys for easy lookup of content.
    The structure is:
    {
        "content_hash": {
            "url": "original_url",
            "links": ["link_folder_name1", "link_folder_name2", ...],
            "depth": depth_at_which_url_was_found,
            "leaf": true_if_no_outgoing_links
        },
        ...
    }
    """
    graph_path = os.path.join(data_folder, "url_graph.json")
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(url_graph, f, indent=2)
    print(f"Saved URL graph to {graph_path}")

def process_html_to_text():
    """Process HTML files to extract their text content."""
    # Get list of folders to process
    folders = [f for f in os.listdir(data_folder) 
               if os.path.isdir(os.path.join(data_folder, f))]
    
    # Check how many need text extraction
    to_process = []
    for folder_name in folders:
        folder_path = os.path.join(data_folder, folder_name)
        html_path = os.path.join(folder_path, "content.html")
        text_path = os.path.join(folder_path, "content.txt")
        metadata_path = os.path.join(folder_path, "metadata.json")
        
        # Skip if it's a duplicate (no content.html)
        if not os.path.exists(html_path) and os.path.exists(metadata_path):
            # Check if it's a duplicate
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if metadata.get("is_duplicate", False):
                    print(f"Skipping duplicate: {folder_name} (duplicate of {metadata.get('duplicate_of', 'unknown')})")
                    continue
        
        # Process only if we have HTML content but no text yet
        if os.path.exists(html_path) and not os.path.exists(text_path):
            to_process.append((folder_name, folder_path, html_path, text_path))
    
    if not to_process:
        print("No HTML files need text extraction.")
        return
    
    print(f"Extracting text from {len(to_process)} HTML files...")
    
    # Process with progress bar
    for folder_name, folder_path, html_path, text_path in tqdm(to_process, desc="Extracting text", unit="file"):
        try:
            # Read HTML and extract text
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
                
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator='\n')
            
            # Remove empty lines and normalize whitespace
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Save text content
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
        except Exception as e:
            print(f"Failed to extract text from {folder_name}: {e}")

def save_data():
    pass

def get_urls_for_depth_extension(current_depth, new_depth):
    """
    Get a list of URLs that should be visited if depth is increased.
    
    Args:
        current_depth (int): The previous max depth
        new_depth (int): The new max depth
        
    Returns:
        list: List of (url, depth) tuples to visit
    """
    # Check if the URL graph exists
    graph_path = os.path.join(data_folder, "url_graph.json")
    if not os.path.exists(graph_path):
        return []
        
    # Load the URL graph
    with open(graph_path, 'r', encoding='utf-8') as f:
        url_graph = json.load(f)
    
    urls_to_visit = []
    
    # Find URLs that were at the frontier of the previous crawl
    for content_hash, info in url_graph.items():
        if info.get('depth') == current_depth and not info.get('leaf', False):
            # This URL was at the maximum depth and has outgoing links
            url = info.get('url')
            if url:
                urls_to_visit.append((url, current_depth))
    
    print(f"Found {len(urls_to_visit)} URLs at depth {current_depth} with outgoing links to explore at depth {new_depth}")
    return urls_to_visit
