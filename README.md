# Endee RAG Assistant  
Semantic Search + RAG-style Document Q&A using Endee Vector Database

---

## Project Overview

This project is a working **AI/ML Document Question Answering application** that lets users upload documents (TXT / PDF), index them using **vector embeddings**, and ask questions using **semantic search**.

It demonstrates how to build a real-world **Retrieval-Augmented Generation (RAG)** workflow where **vector search is the core component**, powered by **Endee** as the vector database.



## Problem Statement

Traditional keyword search fails when:
- users don’t type exact words present in the document
- the same meaning is expressed using different sentences
- documents are large and difficult to read manually

This project solves that by using **semantic similarity search** to retrieve the most relevant text chunks from uploaded documents and answer questions based on that retrieved context.

---

##  System Design / Technical Approach

###  High-Level Pipeline

1. **User uploads documents (PDF/TXT)**  
2. **Text Extraction**
   - TXT is read directly
   - PDF is extracted using `PyPDF2`
3. **Text Cleaning**
   - Removes extra spaces/newlines/symbol noise
4. **Chunking**
   - Splits document into smaller chunks for better retrieval  
   - Uses overlap so context continuity is maintained  
5. **Embedding Generation**
   - Each chunk is converted into an embedding vector using `sentence-transformers`
6. **Vector Storage (Endee)**
   - Vectors + metadata (source, text) are stored inside Endee
7. **Query Search**
   - User query is embedded and searched in Endee using similarity search
8. **Retrieve Top-K Chunks**
   - Most relevant chunks are returned with scores
9. **Answer Generation (RAG-style)**
   - Answer is formed from retrieved context (grounded response)

---

##  System Components

###  1. Streamlit Frontend (UI)
- Upload files
- Index documents
- Ask questions
- Display answers + retrieved chunks

###  2. Chunking & Cleaning (Text Processing)
- `clean_text()` removes unnecessary formatting
- `chunk_text()` splits text into manageable parts

###  3. Embedder (Vectorization)
- Uses Sentence Transformers to convert text into numerical vectors
- These vectors represent meaning and allow semantic comparison

###  4. Vector Database (Endee)
- Stores embeddings efficiently
- Supports similarity search (Top-K nearest vectors)

###  5. Retrieval + Answer Pipeline (RAG)
- Retrieves relevant document sections from Endee
- Generates final answer using retrieved context

---

##  How Endee is Used

Endee is used as the **vector database layer** for this project.

### Endee Responsibilities in this project:
Stores document chunk embeddings  
Stores metadata:
- text chunk
- source file name

Performs similarity search for user queries  
Returns:
- Top-K relevant chunks
- similarity scores (confidence)

This project uses **local mode** for Endee storage:
- vectors are stored in a local folder (ex: `local_db/`)

This makes it easy to run without Docker.

---


---

## Setup Instructions

### Step 1: Create a Virtual Environment
```bash
py -m venv venv

### Step 2: Activate Virtual Environment

venv\Scripts\activate

### Step 3: Install Required Packages

python -m pip install -r requirements.txt

**### Run the Project (Execution Instructions)**

Run Streamlit:

python -m streamlit run app.py


App will open at:

http://localhost:8508



