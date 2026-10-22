from collections import Counter
from pathlib import Path

from src.embedding import MODEL_NAME, load_model
from src.ingestion import load_chunks
from src.retrieval import COLLECTION_NAME, INDEX_PATH, build_index, open_client

KNOWLEDGE_BASE = Path("data/knowledge_base")

def main() -> None:
    """Rebuild the Qdrant index from the Markdown sources."""
    chunks = load_chunks(KNOWLEDGE_BASE)
    client = open_client(INDEX_PATH)
    try:
        build_index(client, load_model(), chunks)
        print(f"Model:      {MODEL_NAME}")
        print(f"Collection: {COLLECTION_NAME} at {INDEX_PATH}")
        print(f"Indexed {client.count(COLLECTION_NAME).count} chunks")
        for document_id, count in Counter(c.document_id for c in chunks).items():
            print(f"  {document_id}: {count}")
    finally:
        client.close()


if __name__ == "__main__":
    main()