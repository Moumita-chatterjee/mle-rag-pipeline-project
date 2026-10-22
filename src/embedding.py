from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ingestion import Chunk, load_chunks

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_model(model_name: str = MODEL_NAME ) -> SentenceTransformer:
    return SentenceTransformer(model_name, device = "cpu")

def chunk_to_text(chunk: Chunk) -> str:
    return f"{chunk.title}.{chunk.text}"

def embedded_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings = True)


if __name__ =="__main__":
    model = load_model()
    chunks = load_chunks(Path("data/knowledge_base"))
    vectors = embedded_texts(model, [chunk_to_text(chunk) for chunk in chunks])
    print("Chunk vectors:", vectors.shape)

    question = "How do I get a refund for a duplicate payment?"
    question_vector = embedded_texts(model, [question])[0]
    scores = vectors @ question_vector

    print(f"\nQuestion: {question}")
    for i in scores.argsort()[::-1][:3]:
        print(f"  {scores[i]:.3f}  {chunks[i].chunk_id}")