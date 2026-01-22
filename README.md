# Agentic RAG

A multi-agent Retrieval-Augmented Generation (RAG) system built with FastAPI and LangChain. This demo showcases how multiple AI agents can work together to provide intelligent responses by retrieving relevant information from a knowledge base, in this case about retail and product information.

## Features

- **Multi-agent architecture**: Separate retriever and responder agents for specialized tasks
- **Vector-based search**: Semantic search using ChromaDB for document retrieval
- **FastAPI REST API**: Clean HTTP endpoints for querying the system
- **Async processing**: Background task processing with webhook callbacks and queue.
- **Configurable**: Environment-based configuration for easy deployment
- **Callback integration**: RAG responses are sent to a configurable callback URL.

## Architecture

The system consists of two main agents:

1. **Retriever Agent**: Performs semantic search in the vector database to find relevant documents.
2. **Responder Agent**: Generates responses using retrieved context and handles user interactions. It is responsible for deciding whether to call the retriever agent or not.

## Demo

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the price and availability of product 1427b3b0?",
       "user_id": "user123"
     }'
```

The result of this question comes from 3 files (product description, prices and stock files). 
The response will be received by the callback endpoint and can be seen in the logs, like this: 
```
INFO:app: * RAG RESPONSE: The product **1427b3b0 (Mystery Book)** is priced at **$111** but is currently **out of stock**.
```

## Prerequisites

- Python 3.12 or higher
- Mistral API key (for the language model)

## Setup
The easiest way to manage the project is using the `make` commands from the `Makefile` file.
Try running this to see all the available commands:

```bash
make help
```

### Option 1: Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic-rag
   ```

2. **Set up environment variables**
   ```bash
   cp .env.sample .env
   ```
   Edit `.env` and add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

3. **Build and start the application**
   ```bash
   docker compose build
   docker compose up
   ```

4. **View logs (optional)**
   ```bash
   docker compose logs -f
   ```

The API will be available at `http://localhost:8000`

### Option 2: Using uv

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic-rag
   ```

2. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Set up environment variables**
   ```bash
   cp .env.sample .env
   ```
   Edit `.env` and add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

### Option 3: Using pip

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic-rag
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   If `requirements.txt` doesn't exist, install from pyproject.toml:
   ```bash
   pip install -e .
   ```

4. **Set up environment variables**
   ```bash
   cp .env.sample .env
   ```
   Edit `.env` and add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## Usage

### Running the Application

#### With Docker:
```bash
docker compose up
```

#### With uv:
```bash
uv run uvicorn app:app --port 8000
```

#### With pip:
```bash
uvicorn app:app --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### API Endpoints

#### POST `/query`
Submit a query to the RAG system.

**Request body:**
```json
{
  "query": "What is the price of Product 1?",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "message": "User query enqueued successfully!"
}
```

#### POST `/webhook`
Receives callback responses (for internal use).


## Configuration

The application can be configured through environment variables in the `.env` file:

- `MISTRAL_API_KEY`: Your Mistral API key (required)
- `DOCUMENTS_DIRECTORY`: Directory containing documents to index (default: "documents")
- `VECTOR_DB_DIRECTORY`: Directory for vector database storage (default: ".vector_db")
- `CHUNK_SIZE`: Text chunk size for document processing (default: 1000)
- `CHUNK_OVERLAP_SIZE`: Overlap between chunks (default: 100)
- `TOP_K`: Number of documents to retrieve (default: 5)
- `CALLBACK_URL`: Webhook URL for responses (optional)

## Document Management

The system automatically indexes documents from the `documents/` directory on startup. Supported formats:
- `.txt` files
- `.md` files

To add new documents:
1. Place files in the `documents/` directory
2. Restart the application to re-index

## Running Tests

#### With Docker:
```bash
docker compose exec app uv run pytest tests/
```

#### With uv:
```bash
uv run pytest tests/
```

#### With pip:
```bash
pytest tests/
```


## Project Structure

```
agentic-rag/
├── agents/           # AI agents (retriever, responder)
├── app/             # FastAPI application
├── documents/       # Knowledge base documents
├── rag/            # RAG implementation and vector store
├── tests/          # Unit tests
├── config.py       # Configuration management
└── pyproject.toml  # Project dependencies
```

## License

This project is licensed under the terms specified in the LICENSE file.
