# Knowledge Graph AI

Knowledge Graph AI is a production-style GraphRAG application that combines:

- Neo4j knowledge graphs
- OpenAI embeddings and LLMs
- Hybrid retrieval
- Graph context expansion
- Streamlit UI
- FastAPI REST API
- Docker
- GitHub Actions CI

## Features

### GraphRAG Question Answering

Ask questions against enterprise documents using:

- Full-text retrieval
- Vector similarity search
- Reciprocal Rank Fusion
- Neo4j graph expansion
- Grounded LLM answers
- Source citations
- Confidence scores

### Document Ingestion

Supported document formats:

- TXT
- PDF
- CSV
- JSON

The ingestion pipeline performs:

```text
Document
   ↓
Text Loading
   ↓
Chunking
   ↓
Entity Extraction
   ↓
Relationship Extraction
   ↓
Neo4j Graph
   ↓
Vector Embeddings