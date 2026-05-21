from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
import pickle
from datetime import datetime
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
data_folder = "data"
chunks_file = "chunks.txt"
faiss_index_file = "faiss_index.bin"
chunks_metadata_file = "chunks_metadata.pkl"

GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables for FAISS
faiss_index = None
chunks_data = []
embeddings_array = None


def read_and_chunk_pdfs(data_folder, chunk_size=500):
    """
    Reads PDF files from the data folder, extracts text, and chunks it into smaller pieces.
    
    Args:
        data_folder (str): Path to the folder containing PDF files.
        chunk_size (int): Maximum number of characters per chunk.
    
    Returns:
        list: A list of text chunks.
    """
    chunks = []
    
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' does not exist")
        return chunks
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_folder, filename)
            print(f"Processing PDF: {filename}")
            
            try:
                with open(filepath, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            # Split text into chunks
                            for i in range(0, len(text), chunk_size):
                                chunk = text[i:i + chunk_size].strip()
                                if chunk:
                                    chunks.append({
                                        "text": chunk,
                                        "file": filename,
                                        "page": page_num + 1
                                    })
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    
    return chunks


def save_chunks_to_file(chunks, filepath=chunks_file):
    """
    Save chunks to a text file for reference and debugging.
    
    Args:
        chunks (list): List of text chunks to save.
        filepath (str): Path to save the chunks file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Chunks saved at: {datetime.now()}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, chunk_info in enumerate(chunks):
            f.write(f"--- CHUNK {i} ---\n")
            f.write(f"File: {chunk_info['file']}, Page: {chunk_info['page']}\n")
            f.write(f"{chunk_info['text']}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"✓ Chunks saved to {filepath}")


def create_faiss_index(chunks, batch_size=50):
    """
    Create FAISS index from chunks and save embeddings and metadata.
    
    Args:
        chunks (list): List of chunk dictionaries containing text, file, and page info.
        batch_size (int): Number of chunks to process in each batch.
    
    Returns:
        tuple: (faiss_index, embeddings_array, chunks_data)
    """
    global faiss_index, chunks_data, embeddings_array
    
    try:
        print("\n" + "="*80)
        print("CREATING FAISS INDEX FROM CHUNKS")
        print("="*80)
        
        if not chunks:
            raise ValueError("No chunks provided")
        
        print(f"\n[Step 1/5] Extracting text from {len(chunks)} chunks...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        print(f"✓ Extracted {len(chunk_texts)} chunk texts")
        
        print(f"\n[Step 2/5] Generating embeddings in batches of {batch_size}...")
        all_embeddings = []
        num_batches = (len(chunk_texts) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(chunk_texts), batch_size), 1):
            batch_texts = chunk_texts[i:i+batch_size]
            print(f"  [Batch {batch_num}/{num_batches}] Embedding chunks {i} to {i+len(batch_texts)}...")
            
            batch_embeddings = embedding_model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"    ✅ Processed batch {batch_num}/{num_batches}")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        print(f"\n✓ Generated embeddings shape: {embeddings_array.shape}")
        
        print(f"\n[Step 3/5] Creating FAISS index...")
        # Create FAISS index (using IndexFlatL2 for L2 distance)
        dimension = embeddings_array.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings_array)
        print(f"✓ FAISS index created with dimension {dimension}")
        print(f"✓ Total vectors in index: {faiss_index.ntotal}")
        
        # Store chunks data
        chunks_data = chunks
        
        print(f"\n[Step 4/5] Saving FAISS index to disk...")
        faiss.write_index(faiss_index, faiss_index_file)
        print(f"✓ FAISS index saved to {faiss_index_file}")
        
        print(f"\n[Step 5/5] Saving chunks metadata...")
        with open(chunks_metadata_file, 'wb') as f:
            pickle.dump(chunks_data, f)
        print(f"✓ Chunks metadata saved to {chunks_metadata_file}")
        
        print("\n" + "="*80)
        print("✓ FAISS INDEX CREATED SUCCESSFULLY")
        print("="*80 + "\n")
        
        return faiss_index, embeddings_array, chunks_data
        
    except Exception as e:
        print(f"\n✗ ERROR creating FAISS index: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def load_faiss_index():
    """
    Load FAISS index and chunks metadata from disk.
    
    Returns:
        tuple: (faiss_index, chunks_data, embeddings_array)
    """
    global faiss_index, chunks_data, embeddings_array
    
    try:
        if not os.path.exists(faiss_index_file):
            print("FAISS index file not found. Please call /create_db first.")
            return None, None, None
        
        print("Loading FAISS index from disk...")
        faiss_index = faiss.read_index(faiss_index_file)
        
        print("Loading chunks metadata...")
        with open(chunks_metadata_file, 'rb') as f:
            chunks_data = pickle.load(f)
        
        print(f"✓ Loaded FAISS index with {faiss_index.ntotal} vectors")
        print(f"✓ Loaded {len(chunks_data)} chunks metadata")
        
        return faiss_index, chunks_data, embeddings_array
        
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
        return None, None, None


def search_similar_chunks(query, top_k=3):
    """
    Search for similar chunks using FAISS index.
    
    Args:
        query (str): User query.
        top_k (int): Number of top similar chunks to return.
    
    Returns:
        list: List of similar chunks with metadata.
    """
    if faiss_index is None or not chunks_data:
        print("ERROR: FAISS index not initialized")
        return []
    
    try:
        print(f"\nSearching for similar chunks (top {top_k})...")
        
        # Generate embedding for query
        query_embedding = embedding_model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS index
        distances, indices = faiss_index.search(query_embedding, top_k)
        
        print(f"✓ Found {len(indices[0])} similar chunks")
        
        # Retrieve chunk data
        similar_chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(chunks_data):
                chunk = chunks_data[int(idx)]
                similar_chunks.append({
                    "index": int(idx),
                    "text": chunk["text"],
                    "file": chunk.get("file", "unknown"),
                    "page": chunk.get("page", "unknown"),
                    "distance": float(distance)
                })
        
        return similar_chunks
        
    except Exception as e:
        print(f"Error searching chunks: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def call_grok_api(prompt):
    """
    Call Grok API to generate a response.
    
    Args:
        prompt (str): Prompt to send to Grok API.
    
    Returns:
        str: Generated response from Grok API.
    """
    if not GROK_API_KEY:
        return "Error: GROK_API_KEY not set in environment variables."
    
    print("\nCalling Grok API...",GROK_API_KEY)
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(GROK_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error calling Grok API: {str(e)}"


# ==================== WEB UI ROUTES ====================

@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def serve_ui():
    """
    Serve the RAG Chatbot UI.
    """
    return render_template("index.html")


# ==================== API ENDPOINTS ====================

@app.route("/create_db", methods=["GET"])
def create_db():
    """
    Create FAISS database from PDF files in the data folder.
    This will:
    1. Read all PDFs from data folder
    2. Extract and chunk text
    3. Generate embeddings
    4. Create and save FAISS index
    """
    try:
        print("\n" + "="*80)
        print("INITIALIZING RAG PIPELINE")
        print("="*80)
        
        # Create data folder if it doesn't exist
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            return jsonify({
                "status": "error",
                "message": f"Data folder created but is empty. Please add PDF files to '{data_folder}' folder."
            }), 400
        
        # Read and chunk PDFs
        print("\n[Stage 1/3] Reading and chunking PDFs...")
        chunks = read_and_chunk_pdfs(data_folder)
        
        if not chunks:
            return jsonify({
                "status": "error",
                "message": f"No PDF files found in '{data_folder}' folder"
            }), 400
        
        print(f"✓ Extracted {len(chunks)} chunks from PDFs")
        
        # Save chunks to file
        print("\n[Stage 2/3] Saving chunks to text file...")
        save_chunks_to_file(chunks)
        
        # Create FAISS index
        print("\n[Stage 3/3] Creating FAISS index...")
        create_faiss_index(chunks)
        
        return jsonify({
            "status": "success",
            "message": "RAG pipeline initialized successfully",
            "total_chunks": len(chunks),
            "chunks_saved_to": chunks_file,
            "faiss_index_file": faiss_index_file,
            "metadata_file": chunks_metadata_file
        }), 200
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/get_query_answer", methods=["POST"])
def get_query_answer():
    """
    Process user query and return answer using RAG pipeline.
    
    Expected request body:
    {
        "query": "Your question here"
    }
    
    Returns:
    {
        "response": "Generated answer from LLM",
        "chunks_used": [
            {
                "text": "Relevant chunk text",
                "file": "source_file.pdf",
                "page": 1
            },
            ...
        ],
        "status": "success"
    }
    """
    try:
        user_query = request.json.get("query")
        
        if not user_query:
            return jsonify({
                "status": "error",
                "message": "Query is required"
            }), 400
        
        print(f"\n{'='*80}")
        print(f"Processing query: {user_query}")
        print(f"{'='*80}")
        
        # Load FAISS index if not already loaded
        if faiss_index is None:
            print("\nLoading FAISS index...")
            load_faiss_index()
        
        if faiss_index is None:
            return jsonify({
                "status": "error",
                "message": "FAISS database not initialized. Please call /create_db first"
            }), 400
        
        # Search for similar chunks
        print("\n[Step 1/3] Searching for similar chunks...")
        similar_chunks = search_similar_chunks(user_query, top_k=3)
        
        if not similar_chunks:
            return jsonify({
                "status": "error",
                "message": "No relevant documents found"
            }), 404
        
        print(f"✓ Found {len(similar_chunks)} relevant chunks")
        
        # Build context from retrieved chunks
        print("\n[Step 2/3] Building context from retrieved chunks...")
        context = "\n\n".join([chunk["text"] for chunk in similar_chunks])
        print(f"✓ Context prepared ({len(context)} characters)")
        
        # Prepare prompt for LLM
        print("\n[Step 3/3] Generating response using Grok API...")
        prompt = f"""Based on the following context from PDF documents, answer the user's question comprehensively.

CONTEXT:
{context}

USER QUESTION: {user_query}

Please provide a detailed answer based on the provided context."""
        
        # Call Grok API
        response = call_grok_api(prompt)
        
        print(f"\n{'='*80}")
        print("✓ Query processed successfully")
        print(f"{'='*80}")
        
        # Prepare chunks metadata for response
        chunks_used = [
            {
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "file": chunk["file"],
                "page": chunk["page"],
                "distance": chunk["distance"]
            }
            for chunk in similar_chunks
        ]
        
        return jsonify({
            "status": "success",
            "response": response,
            "chunks_used": chunks_used,
            "total_chunks_searched": len(chunks_data)
        }), 200
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """
    return jsonify({"status": "healthy"}), 200


@app.route("/status", methods=["GET"])
def status():
    """
    Get the status of the RAG pipeline.
    """
    try:
        if faiss_index is None:
            return jsonify({
                "status": "not_initialized",
                "message": "RAG pipeline not initialized. Call /create_db first",
                "faiss_index_exists": os.path.exists(faiss_index_file)
            }), 200
        
        return jsonify({
            "status": "ready",
            "message": "RAG pipeline is ready",
            "total_chunks": len(chunks_data),
            "faiss_index_size": faiss_index.ntotal,
            "embedding_dimension": embeddings_array.shape[1] if embeddings_array is not None else "unknown",
            "chunks_file": chunks_file if os.path.exists(chunks_file) else "not found",
            "faiss_index_file": faiss_index_file if os.path.exists(faiss_index_file) else "not found",
            "metadata_file": chunks_metadata_file if os.path.exists(chunks_metadata_file) else "not found"
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    print("="*80)
    print("RAG PIPELINE WITH FAISS DB")
    print("="*80)
    print("\n🌐 Web UI:")
    print("  - http://127.0.0.1:5000/")
    print("  - http://127.0.0.1:5000/index.html")
    print("\n📡 Available API endpoints:")
    print("  - GET  /create_db - Initialize RAG database from PDFs")
    print("  - POST /get_query_answer - Query the RAG pipeline")
    print("  - GET  /status - Check pipeline status")
    print("  - GET  /health - Health check")
    print("\n📋 Setup Instructions:")
    print("  1. Place PDF files in the 'data' folder")
    print("  2. Open http://127.0.0.1:5000 in your browser")
    print("  3. Click 'Initialize Database' to build the FAISS index")
    print("  4. Start asking questions!")
    print("="*80)
    
    # Auto-load FAISS index if it exists
    print("\n🔍 Checking for existing FAISS index...")
    if os.path.exists(faiss_index_file) and os.path.exists(chunks_metadata_file):
        print("✓ Found existing FAISS index. Loading...")
        try:
            load_faiss_index()
            print("✓ Database loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading database: {str(e)}")
    else:
        print("✗ No existing FAISS index found.")
        print("  Please click 'Initialize Database' in the UI to create one.\n")
    
    app.run(debug=True)
