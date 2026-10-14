import json
import numpy as np
import onnxruntime as ort
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoTokenizer


class FastEmbedder:

    def __init__(self, model_dir: str = "onnx_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(f"{model_dir}/model.onnx")

    def encode(self, text: str) -> list[float]:
        inputs = self.tokenizer(
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

        # Run ONNX Inference
        outputs = self.session.run(None, ort_inputs)

        # Mean pooling
        last_hidden_state = outputs[0]
        input_mask_expanded = np.expand_dims(inputs["attention_mask"], -1)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embedding = (sum_embeddings / sum_mask)[0]

        # Normalization
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist()


def build_vector_database():
    # 1. Initialize Qdrant lokal (disimpan dalam folder 'qdrant_db')
    client = QdrantClient(path="qdrant_db")
    collection_name = "financial_docs"

    print(" Initializing Qdrant Collection...")
    # Model all-MiniLM-L6-v2 menghasilkan vektor berdimensi 384
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # 2. Load JSON Docs
    with open("financial_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    embedder = FastEmbedder()
    points = []

    print(f" Embedding {len(docs)} documents via ONNX and indexing...")
    for doc in docs:
        vector = embedder.encode(doc["text"])
        points.append(
            PointStruct(
                id=doc["id"],
                vector=vector,
                payload={"text": doc["text"], "sentiment": doc["sentiment"]},
            )
        )

    # 3. Upsert ke Qdrant
    client.upsert(collection_name=collection_name, points=points)
    print(f" Successfully indexed {len(points)} documents into Qdrant Vector DB!")


if __name__ == "__main__":
    build_vector_database()