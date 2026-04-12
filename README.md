# Retrieval-Augmented Generation (RAG) Pipeline with FAISS

A complete RAG (Retrieval-Augmented Generation) application built using Python and Flask. This application processes PDF files, stores embeddings in FAISS (Facebook AI Similarity Search), and uses the Grok API for generating intelligent responses.

## Features

- **PDF Processing**: Automatically reads and chunks PDF files from the `data` folder
- **FAISS Indexing**: Uses Facebook AI Similarity Search for fast and efficient vector similarity search
- **Free Embeddings**: Uses `sentence-transformers` library with the `all-MiniLM-L6-v2` model
- **Grok API Integration**: Generates responses using Grok API based on retrieved context
- **Batch Processing**: Efficiently processes large numbers of chunks in batches
- **Persistent Storage**: Saves FAISS index and chunk metadata to disk for reuse

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd dm_rag
```

### 2. Create a virtual environment
```bash
python -m venv dm_rag_env
source dm_rag_env/bin/activate  # On Windows: dm_rag_env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Grok API Key
Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` and add your Grok API key:
```
GROK_API_KEY=your_actual_api_key_here
```

### 5. Add PDF files
Place your PDF files in the `data` folder:
```
dm_rag/
├── data/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
```

### 6. Run the application
```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

## API Endpoints

### 1. `/create_db` (GET)
**Purpose**: Initialize the RAG pipeline by reading PDFs and creating the FAISS index.

**Request**:
```bash
curl http://127.0.0.1:5000/create_db
```

**Response**:
```json
{
  "status": "success",
  "message": "RAG pipeline initialized successfully",
  "total_chunks": 150,
  "chunks_saved_to": "chunks.txt",
  "faiss_index_file": "faiss_index.bin",
  "metadata_file": "chunks_metadata.pkl"
}
```

**What it does**:
1. Reads all PDF files from the `data` folder
2. Extracts and chunks text into smaller pieces
3. Saves chunks to `chunks.txt` for reference
4. Generates embeddings for each chunk
5. Creates and saves FAISS index to disk
6. Saves chunk metadata for later retrieval

---

### 2. `/get_query_answer` (POST)
**Purpose**: Query the RAG pipeline and get an intelligent response from the LLM.

**Request**:
```bash
curl -X POST http://127.0.0.1:5000/get_query_answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the documents?"
  }'
```

**Response**:
```json
{
  "status": "success",
  "response": "Based on the provided documents, the main topic is...",
  "chunks_used": [
    {
      "text": "Relevant excerpt from document...",
      "file": "document1.pdf",
      "page": 2,
      "distance": 0.45
    },
    {
      "text": "Another relevant excerpt...",
      "file": "document2.pdf",
      "page": 5,
      "distance": 0.52
    }
  ],
  "total_chunks_searched": 150
}
```

**What it does**:
1. Takes the user query
2. Generates embedding for the query
3. Searches FAISS index for 3 most similar chunks
4. Builds context from retrieved chunks
5. Sends context and query to Grok API
6. Returns the generated response along with the chunks used

---

### 3. `/status` (GET)
**Purpose**: Check the status of the RAG pipeline.

**Request**:
```bash
curl http://127.0.0.1:5000/status
```

**Response** (when initialized):
```json
{
  "status": "ready",
  "message": "RAG pipeline is ready",
  "total_chunks": 150,
  "faiss_index_size": 150,
  "embedding_dimension": 384,
  "chunks_file": "chunks.txt",
  "faiss_index_file": "faiss_index.bin",
  "metadata_file": "chunks_metadata.pkl"
}
```

---

### 4. `/health` (GET)
**Purpose**: Health check endpoint.

**Request**:
```bash
curl http://127.0.0.1:5000/health
```

**Response**:
```json
{
  "status": "healthy"
}
```

---

## How the RAG Pipeline Works

### Step 1: Database Creation (`/create_db`)
1. **Read PDFs**: Scans the `data` folder for PDF files
2. **Extract Text**: Uses PyPDF2 to extract text from each PDF page
3. **Chunking**: Splits extracted text into 500-character chunks
4. **Batch Embedding**: Generates embeddings in batches of 50 chunks
5. **FAISS Indexing**: Creates a FAISS index for fast similarity search
6. **Persistence**: Saves index and metadata to disk

### Step 2: Query Processing (`/get_query_answer`)
1. **Query Embedding**: Generates embedding for user's question
2. **Similarity Search**: Searches FAISS index for 3 most similar chunks
3. **Context Building**: Combines retrieved chunks into a context
4. **LLM Prompt**: Creates a prompt with context and query
5. **API Call**: Sends to Grok API for response generation
6. **Response**: Returns both the answer and the chunks used

## File Structure

```
dm_rag/
├── app.py                      # Main Flask application
├── data/                        # Place your PDF files here
├── chunks.txt                   # Extracted chunks (created after /create_db)
├── faiss_index.bin              # FAISS index (created after /create_db)
├── chunks_metadata.pkl          # Chunk metadata (created after /create_db)
├── .env.example                 # Example environment configuration
├── .gitignore                   # Git ignore file
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## Dependencies

| Package | Purpose |
|---------|---------|
| **flask** | Web framework for API endpoints |
| **PyPDF2** | PDF file parsing and text extraction |
| **faiss-cpu** | Vector similarity search (FAISS) |
| **sentence-transformers** | Free embedding model (all-MiniLM-L6-v2) |
| **requests** | HTTP requests for Grok API |
| **python-dotenv** | Load environment variables from .env |
| **numpy** | Numerical computing for embeddings |

## Configuration

### Batch Size for Embedding Generation
To adjust batch size for better performance, modify in `app.py`:
```python
create_faiss_index(chunks, batch_size=100)  # Larger batches for more RAM
create_faiss_index(chunks, batch_size=25)   # Smaller batches for less RAM
```

### Number of Chunks to Retrieve
To change how many similar chunks are retrieved for context, modify in `app.py`:
```python
similar_chunks = search_similar_chunks(user_query, top_k=5)  # Get top 5 instead of 3
```

## Example Usage Workflow

```bash
# 1. Create the RAG database
curl http://127.0.0.1:5000/create_db

# 2. Check status
curl http://127.0.0.1:5000/status

# 3. Query the system
curl -X POST http://127.0.0.1:5000/get_query_answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the summary of the documents?"}'

# 4. Ask another question
curl -X POST http://127.0.0.1:5000/get_query_answer \
  -H "Content-Type: application/json" \
  -d '{"query": "Can you explain the key concepts?"}'
```

## License

This project is licensed under the MIT License.