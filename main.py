import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from prometheus_client import Histogram, make_asgi_app
from pydantic import BaseModel
from qdrant_client import QdrantClient
from transformers import AutoTokenizer

app = FastAPI(title="Financial RAG & Sentiment Microservice")

# Prometheus Metric untuk Latency Observability
SEARCH_LATENCY = Histogram(
    "rag_search_latency_seconds",
    "Time spent embedding query and searching Qdrant DB",
)

MODEL_DIR = "onnx_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
onnx_session = ort.InferenceSession(f"{MODEL_DIR}/model.onnx")
qdrant_client = QdrantClient(path="qdrant_db")
COLLECTION_NAME = "financial_docs"


def encode_query(text: str) -> list[float]:
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    outputs = onnx_session.run(None, ort_inputs)

    last_hidden_state = outputs[0]
    input_mask_expanded = np.expand_dims(inputs["attention_mask"], -1)
    sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    embedding = (sum_embeddings / sum_mask)[0]

    norm = np.linalg.norm(embedding)
    return (embedding / norm).tolist()


class QueryPayload(BaseModel):
    query: str
    top_k: int = 3


@app.post("/api/v1/search")
def search_financial_docs(payload: QueryPayload):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time()

    try:
        # 1. Embed Query via ONNX
        query_vector = encode_query(payload.query)

        # 2. Vector Search ke Qdrant (Dukungan Kompatibilitas Versi Qdrant)
        if hasattr(qdrant_client, "query_points"):
            res = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=payload.top_k,
            )
            search_result = res.points
        else:
            search_result = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=payload.top_k,
            )

        latency = time.time() - start_time
        SEARCH_LATENCY.observe(latency)

        results = []
        for hit in search_result:
            results.append(
                {
                    "score": round(hit.score, 4),
                    "text": hit.payload["text"],
                    "sentiment": hit.payload["sentiment"],
                }
            )

        return {
            "query": payload.query,
            "latency_ms": round(latency * 1000, 2),
            "results_count": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Execution Error: {str(e)}")


# Mount Endpoint Prometheus /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)