# OCF Assistant

Current technologies allow RAG to greatly improve the quality of a chatbot. This system integrates ollama, qdrant, and scraping/parsing to create a chatbot that can answer questions about the OCF.

## Scraping

The webscraping takes a base url and safe patterns to search via config.yml. It then crawls the web, extracting text from the pages and saving the url graph to a file. It's built so that you can run the program many times, and it will resume from where it left off, not performing duplicate fetching. (Idempotency)

TODO: if the safe-urls list is appended to, will the system be able to crawl the new pages?

## Chunking

Check embeddings.py for the code. Essentially, use NTLK or similar semantic parsing in a hybrid system with naive chunking to create chunks for embeddings.

## Embeddings (parsing)

Uses snowflake-arctic-embed2 via ollama to generate embeddings. Idempotent.

## Chat

Uses llama3.2:3b via ollama to generate responses. 

RAG is performed at 10 responses, and naively fed into the chat pipeline.

## Web Interface

The OCF Assistant includes a modern web interface that can be accessed by running:

```
python main.py --web
```

This starts a web server on port 8080 that provides a user-friendly interface for:
- Chatting with the OCF Assistant
- Managing data operations (scraping, chunking, embeddings)
- Configuring system settings

The web interface runs on Flask and provides a convenient way to interact with the OCF Assistant without using the command line.

### WebSocket Integration

The interface includes WebSocket support with the following features:
- Real-time connectivity status with automatic reconnection
- Ping-pong mechanism for connection monitoring
- Live system metrics dashboard
- Connection status displays in the UI

The WebSocket implementation provides:
- Automatic reconnection with exponential backoff
- Real-time status updates without page refreshes
- A dashboard for monitoring system metrics

TODO:
- reranking
- more sophistocated hybrid search
- pregenerating nearby search terms for better document retrieval results
- semantic search
- API endpoints for the web interface to communicate with the backend


