#! /bin/bash

# Process command-line arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "OCF Assistant Runner Script"
  echo "Usage: ./run.sh [command]"
  echo ""
  echo "Commands:"
  echo "  --scrape           Run web scraping"
  echo "  --chunk            Run text chunking"
  echo "  --embeddings       Generate embeddings for chunks"
  echo "  --qdrant           Upload embeddings to Qdrant"
  echo "  --chat             Start chat interface with RAG"
  echo "  --pipeline         Run full pipeline (scrape + chunk + embeddings + qdrant)"
  echo "  --help, -h         Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./run.sh --scrape  # Run web scraping with default parameters"
  echo "  ./run.sh --chat    # Start the chat interface"
  exit 0
fi

# Handle different commands
case "$1" in
  --scrape)
    echo "Running web scraping..."
    poetry run python main.py --scrape --depth 2 --extend-depth
    ;;
  --chunk)
    echo "Running text chunking..."
    poetry run python main.py --chunk
    ;;
  --embeddings)
    echo "Generating embeddings for chunks..."
    poetry run python main.py --embeddings
    ;;
  --qdrant)
    echo "Uploading embeddings to Qdrant..."
    poetry run python main.py --qdrant
    ;;
  --chat)
    echo "Starting chat interface..."
    poetry run python main.py --chat
    ;;
  --pipeline)
    echo "Running full pipeline..."
    echo "Step 1: Web scraping"
    poetry run python main.py --scrape --depth 2 --extend-depth
    
    echo "Step 2: Text chunking"
    poetry run python main.py --chunk
    
    echo "Step 3: Generating embeddings"
    poetry run python main.py --embeddings
    
    echo "Step 4: Uploading to Qdrant"
    poetry run python main.py --qdrant
    
    echo "Pipeline complete! You can now use --chat to interact with the assistant."
    ;;
  *)
    echo "Running web server..."
    poetry run python main.py --web
    ;;
esac
