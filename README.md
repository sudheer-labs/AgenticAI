# AEGIS – Advanced Enterprise RAG System

## Overview

AEGIS is an enterprise-grade Retrieval-Augmented Generation (RAG) platform designed for intelligent document retrieval, semantic search, and AI-powered conversational querying.

The system enables organizations to:

- Ingest enterprise documents
- Generate embeddings
- Store vectors in a vector database
- Retrieve context-aware information
- Interact with documents using LLM-powered chat interfaces

This project is built for scalable AI applications using modern Python-based architectures.

---

# Features

- Enterprise Document Ingestion Pipeline
- Multi-file Knowledge Base Support
- Semantic Search with Vector Embeddings
- FastAPI-based Backend APIs
- AI Chat Interface Integration
- Modular Source Code Architecture
- Batch Embedding Processing
- Extensible LLM Integration
- Scalable RAG Workflow Design
- Production Deployment Ready

---

# Live Deployment

## UI Application

Render frontend URL:

```text
https://aegis-policy-ui.onrender.com
```

---

## Backend API

Render backend API URL:

```text
https://aegis-policy-api.onrender.com
```

---

# Project Structure

```bash
AgenticAI/
│
├── ingestion/
│   ├── embedding.py
│   ├── chunking.py
│   ├── loaders.py
│   └── preprocessing.py
│
├── retrieval/
│   ├── hybrid_search.py
│   ├── reranker.py
│   └── query_engine.py
│
├── prompts/
│   ├── system_prompts.py
│   └── templates.py
│
├── utils/
│   ├── helpers.py
│   ├── logger.py
│   └── config.py
│
├── static/
│   └── uploads/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── api.py                      # FastAPI application
├── main.py                     # Main entrypoint
├── app.py                      # UI/Frontend launcher
├── requirements.txt
├── render.yaml                 # Render deployment config
├── Dockerfile
├── .env
├── .gitignore
└── README.md
```

---

# Architecture

```text
Documents
    ↓
Ingestion Pipeline
    ↓
Chunking & Embedding Generation
    ↓
Vector Database Storage
    ↓
Semantic Retrieval
    ↓
LLM Context Augmentation
    ↓
AI Response Generation
```

---

# Tech Stack

## Backend
- Python
- FastAPI

## AI/LLM
- OpenAI
- LangChain
- RAG Architecture

## Vector Database
- Weaviate

## Frontend/UI
- Streamlit / Custom UI

## Cloud & Deployment
- Render

---

# Installation

## Clone Repository

```bash
git clone https://github.com/sudheer-labs/AgenticAI.git
cd AgenticAI
```

---

# Create Virtual Environment

## Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Linux/Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
```

---

# Run Document Ingestion

```bash
python run_ingestion.py
```

This will:

- Read documents
- Chunk content
- Generate embeddings
- Push vectors to the vector database

---

# Run API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

API will run on:

```text
http://127.0.0.1:8000
```

---

# Run Application UI

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

# Example Workflow

## Step 1
Upload enterprise documents

## Step 2
Run ingestion pipeline

## Step 3
Generate embeddings

## Step 4
Store vectors in Weaviate

## Step 5
Query documents through chatbot/API

---

# Sample API Request

```python
import requests

payload = {
    "query": "What is the leave policy?"
}

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json=payload
)

print(response.json())
```

---

# Deployment

The project can be deployed using:

- Render
- AWS EC2
- AWS EKS
- Docker
- Kubernetes

---

# Render Deployment Notes

## Frontend Service
- Deploy as a Web Service
- Add environment variables in Render dashboard
- Expose frontend port correctly

## Backend Service
- Deploy FastAPI service
- Start command:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

- Configure health checks
- Add required API keys

---

# Security Considerations

- Store API keys securely using environment variables
- Enable authentication for production APIs
- Restrict vector database access
- Implement request validation and rate limiting

---

# Future Enhancements

- Multi-agent orchestration
- Hybrid search support
- Role-based access control
- Conversation memory
- Source citation tracking
- Streaming responses
- Kubernetes deployment support
- Observability and monitoring

---

# Contributing

Contributions are welcome.

## Steps

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Create a Pull Request

---

# License

This project is licensed under the MIT License.

---

# Author

Sudheer Labs

GitHub:
https://github.com/sudheer-labs

---

# Repository

https://github.com/sudheer-labs/AgenticAI
