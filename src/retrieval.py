import uuid
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.embedding import chunk_to_text, embedded_texts, load_model
from src.ingestion import Chunk, load_chunks

COLLECTION_NAME = "northstar_kb"


def build_index(
    client: QdrantClient, model: SentenceTransformer, chunks: list[Chunk]
) -> None:
    """Embed every chunk and (re)create the Qdrant collection from scratch."""
    vectors = embedded_texts(model, [chunk_to_text(chunk) for chunk in chunks])

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
            vector=vector.tolist(),
            payload=asdict(chunk),
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(
    client: QdrantClient, model: SentenceTransformer, question: str, top_k: int = 3
) -> list[tuple[float, Chunk]]:
    """Return the top_k most similar chunks as (score, chunk) pairs."""
    question_vector = embedded_texts(model, [question])[0]
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_vector.tolist(),
        limit=top_k,
    )
    return [(point.score, Chunk(**point.payload)) for point in response.points]


if __name__ == "__main__":
    client = QdrantClient(":memory:")
    model = load_model()
    chunks = load_chunks(Path("data/knowledge_base"))
    build_index(client, model, chunks)
    print("Points in collection:", client.count(COLLECTION_NAME).count)

    question = "How do I get a refund for a duplicate payment?"
    print(f"\nQuestion: {question}")
    for score, chunk in search(client, model, question):
        print(f"  {score:.3f}  {chunk.chunk_id}")