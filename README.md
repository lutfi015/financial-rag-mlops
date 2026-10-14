# Production-Ready Financial RAG & Sentiment Microservice

A high-performance, low-latency **Retrieval-Augmented Generation (RAG)** and financial sentiment analysis microservice built with **Python**, **FastAPI**, **ONNX Runtime**, **Qdrant Vector DB**, and **Prometheus**.

This service is optimized for CPU inference, reaching **sub-10ms latency** by converting Hugging Face Sentence Transformers into ONNX format for efficient vector retrieval and sentiment context extraction.

---

## Key Features & MLOps Architecture

- **Low-Latency ONNX Inference**: Exported and optimized `sentence-transformers/all-MiniLM-L6-v2` to **ONNX Runtime**, reducing vector calculation latency to **<10ms** on CPU without relying on expensive GPU instances.
- **Local Vector Search**: Integrated with **Qdrant Vector DB** utilizing Cosine distance metric for fast semantic search over financial phrase datasets.
- **Production RESTful API**: Built using **FastAPI** with robust request validation (Pydantic), error handling, and structured JSON output.
- **Observability & Monitoring**: Exposed custom Prometheus metrics (`/metrics`) tracking end-to-end RAG retrieval latency (`rag_search_latency_seconds`).
- **Production Packaging**: Fully containerized with **Docker** and **Docker Compose** for reproducible deployment across cloud environments.

---

## Tech Stack

- **Language**: Python 3.13
- **Framework**: FastAPI, Uvicorn, Pydantic
- **ML / Vector DB**: ONNX Runtime, Hugging Face Transformers, Qdrant Client
- **Observability**: Prometheus Client
- **Containerization**: Docker, Docker Compose

---

## Project Structure

```text
financial-rag-mlops/
├── onnx_model/            # Exported ONNX model & Tokenizer artifacts
├── qdrant_db/             # Embedded Qdrant Vector Storage
├── .gitignore             # Python & environment ignore file
├── Dockerfile             # Multi-stage production Docker blueprint
├── docker-compose.yml     # Container orchestration configuration
├── export_onnx.py         # Script to convert PyTorch model -> ONNX format
├── financial_docs.json    # Processed financial phrase dataset
├── index_qdrant.py        # Vector embedding and Qdrant ingestion pipeline
├── ingest.py              # Dataset preparation pipeline
├── main.py                # FastAPI application & Prometheus instrumentation
├── README.md              # Documentation
└── requirements.txt       # Production dependencies
```

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed and a virtual environment set up:

```bash
python -m venv .venv
# On Windows
.\.venv\Scripts\Activate.ps1
# On Linux/macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pipeline Execution (Data Preparation to Ingestion)

Run the following scripts in order to generate data, export the ONNX model, and index vectors into Qdrant:

```bash
# Step 1: Generate dataset
python ingest.py

# Step 2: Export Sentence Transformer to ONNX
python export_onnx.py

# Step 3: Embed documents & build Qdrant vector index
python index_qdrant.py
```

### 4. Run the API Microservice
```bash
uvicorn main:app --reload --port 8000
```

Access the interactive API documentation at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Prometheus Metrics**: [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

---

## Running with Docker

Deploy the microservice into an isolated container using Docker Compose:

```bash
docker compose up --build -d
```

Stop the service:
```bash
docker compose down
```

---

## Sample Request & Response

### API Endpoint
`POST /api/v1/search`

### Request Payload
```json
{
  "query": "operating profit increased",
  "top_k": 3
}
```

### Response Output (Sub-10ms Benchmark)
```json
{
  "query": "operating profit increased",
  "latency_ms": 7.29,
  "results_count": 3,
  "results": [
    {
      "score": 0.6604,
      "text": "Operating profit rose to EUR 12.5 million from EUR 8.7 million in the corresponding period. (Record #93)",
      "sentiment": "positive"
    },
    {
      "score": 0.6595,
      "text": "Operating profit rose to EUR 12.5 million from EUR 8.7 million in the corresponding period. (Record #85)",
      "sentiment": "positive"
    },
    {
      "score": 0.658,
      "text": "Operating profit rose to EUR 12.5 million from EUR 8.7 million in the corresponding period. (Record #21)",
      "sentiment": "positive"
    }
  ]
}
```