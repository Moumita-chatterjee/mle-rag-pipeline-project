from pathlib import Path
from src.ingestion import chunk_document, load_chunks, split_paragraph

KNOWLEDGE_BASE = Path("data/knowledge_base")

def test_split_paragraph_joins_wrapped_lines():
    body = "First line\nstill first paragraph.\n\nSecond paragraph."
    assert split_paragraph(body) == ["First line still first paragraph.", "Second paragraph."]

def test_chunk_document_keeps_metadata(tmp_path):
    path = tmp_path/"example-doc.md"
    path.write_text("# Example Title\n\nFirst paragraph. \n\nSecond paragraph. \n", encoding = "utf-8")

    chunks = chunk_document(path)

    assert len(chunks) == 2
    assert chunks[1].chunk_id == "example-doc:1"
    assert chunks[1].document_id == "example-doc"
    assert chunks[1].title == "Example Title"
    assert chunks[1].chunk_index == 1
    assert chunks[1].text == "Second paragraph."

def test_knowledge_base_chunks_have_unique_ids():
    chunks = load_chunks(KNOWLEDGE_BASE)
    chunks_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunks) == 22
    assert len(set(chunks_ids)) == len(chunks_ids)