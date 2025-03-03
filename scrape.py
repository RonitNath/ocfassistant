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

# Get the config
with open('config.yml', 'r') as f:
    from yaml import safe_load

    config = safe_load(f)

data_folder = config['data_folder']
scrape_safe_patterns = config['scrape_safe_patterns']
scrape_root = config['scrape_root']

def scraper(depth=1):
    """
    Use scrape_root to get the intiial document
    Get all the links from the initial document, and filter them using scrape_safe_patterns

    For each page of data, compute the hash 6-bit hash of the page
    If the hash is not in the data_folder, save the page to the data_folder, titled "safe-url-<hash>.html"
    If the hash is in the data_folder, skip the page

    Create a graph of the links between the data in json format, with the following schema:
    {
        "url": "<sanitized-url-hash>",
        "links": ["<sanitized-url-hash>", "<sanitized-url-hash>", ...]
    }

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
    visited_urls = set()
    urls_to_visit = [(url, 0) for url in scrape_root]  # (url, depth)

    
    # Dictionary to map URLs to their folder names
    url_to_folder = {}
    
    # For the progress bar, we'll use an estimated number of URLs to visit
    # This will adjust as we discover more URLs
    estimated_urls = len(urls_to_visit) * (depth + 1)  # Simple estimation
    
    # Stage 1: URL crawling with progress bar
    print(f"Stage 1: Crawling URLs (initial horribly inaccurate estimate: {estimated_urls} URLs)...")
    pbar = tqdm(total=estimated_urls, desc="Crawling", unit="URL")
    
    processed_count = 0
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
            if not content:
                continue
                
            url_hash = hash_url(current_url)
            sanitized_url = sanitize_url(current_url)
            folder_name = f"{sanitized_url}-{url_hash}"
            
            # Store mapping of URL to folder name
            url_to_folder[current_url] = folder_name
            
            save_html_content(url_hash, current_url, content)
            
            # Extract and process links if not at max depth
            if current_depth < depth:
                links = extract_links(current_url, content)
                link_folders = []
                
                # If we found new links, update our estimated total
                new_links = [link for link in links if link not in visited_urls and is_url_safe(link)]
                if new_links:
                    pbar.total += len(new_links)
                    pbar.refresh()
                
                for link in links:
                    if link not in visited_urls and is_url_safe(link):
                        link_hash = hash_url(link)
                        link_sanitized = sanitize_url(link)
                        link_folder = f"{link_sanitized}-{link_hash}"
                        link_folders.append(link_folder)
                        # Add to queue for next depth level
                        urls_to_visit.append((link, current_depth + 1))
                
                # Add to graph
                url_graph[folder_name] = {
                    "url": current_url,
                    "links": link_folders
                }
            else:
                # No outgoing links at max depth
                url_graph[folder_name] = {
                    "url": current_url,
                    "links": []
                }
                
            # Sleep to be nice to the server
            # time.sleep(0.5) haha jk
            
        except Exception as e:
            pbar.write(f"Error processing {current_url}: {e}")
    
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

def hash_url(url):
    """Create a 6-character hash of a URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:6]

def fetch_url(url):
    """Fetch the content of a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f" Failed to fetch {url}: {e}")
        # add failed url to failed_urls.txt, don't duplicate    
        with open("failed_urls.txt", "a") as f:
            if url not in f.read():
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

def save_html_content(url_hash, original_url, content):
    """Save HTML content to a file."""
    # Create a sanitized version of the URL for the folder name
    sanitized_url = sanitize_url(original_url)
    folder_name = f"{sanitized_url}-{url_hash}"
    
    # Create directory for the folder if it doesn't exist
    hash_dir = os.path.join(data_folder, folder_name)
    if not os.path.exists(hash_dir):
        os.makedirs(hash_dir)
    
    # Save the HTML content
    html_path = os.path.join(hash_dir, "content.html")
    if not os.path.exists(html_path):
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Save original URL for reference
        with open(os.path.join(hash_dir, "url.txt"), 'w', encoding='utf-8') as f:
            f.write(original_url)
    else:
        pass  # Skip silently as we're using progress bar now

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
    """Save the URL graph to a JSON file."""
    graph_path = os.path.join(data_folder, "url_graph.json")
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(url_graph, f, indent=2)
    print(f"Saved URL graph to {graph_path}")

def process_html_to_text():
    """Process HTML files to extract their text content."""
    # Get list of folders to process
    folders = [f for f in os.listdir(data_folder) 
               if os.path.isdir(os.path.join(data_folder, f)) and f != "url_graph.json"]
    
    # Check how many need text extraction
    to_process = []
    for folder_name in folders:
        folder_path = os.path.join(data_folder, folder_name)
        html_path = os.path.join(folder_path, "content.html")
        text_path = os.path.join(folder_path, "content.txt")
        
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
