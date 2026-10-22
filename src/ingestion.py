from pathlib import Path
from dataclasses import dataclass
from collections import Counter

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    chunk_index:int
    text: str 

def list_documents(folder:Path) -> list[Path]:
    list_paths = sorted(folder.glob("*.md"))
    return list_paths

def read_document(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    first_line, _ , body = text.partition("\n")
    title = first_line.removeprefix("#").strip()
    return title, body.strip()

def split_paragraph(body: str) -> list[str]:
    paragraphs = []
    for block in body.split("\n\n"):
        paragraph = " ".join(block.split())
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs

def chunk_document(path:Path) -> list[Chunk]:
    document_id = path.stem
    title, body = read_document(path)
    chunks = []
    for index, paragraph in enumerate(split_paragraph(body)):
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}:{index}",
                document_id=document_id,
                title=title,
                chunk_index=index,
                text=paragraph,
            )
        )
    return chunks

def load_chunks(folder: Path) -> list[Chunk]:
    """Load every Markdown document in the folder as paragraph chunks."""
    chunks = []
    for path in list_documents(folder):
        chunks.extend(chunk_document(path))
    return chunks


if __name__=="__main__":
    knowledge_folder = Path("data/knowledge_base")
    paths = list_documents(knowledge_folder)

    print(f"found {len(paths)} documents")
    

    # title, body = read_document(paths[2])
    # paragraphs = split_paragraph(body)
    # print("title:",title)
    # print(f"{len(paragraphs)} paragraphs")
    # for i, paragraph in enumerate(paragraphs):
    #     print(f"[{i}] ({len(paragraph)} chars) {paragraph}")
   
    for chunk in chunk_document(paths[2]):
        print(chunk)

    chunks = load_chunks(knowledge_folder)

    print(f"Total chunks: {len(chunks)}")
    counts = Counter(chunk.document_id for chunk in chunks)
    for document_id, count in counts.items():
        print(f"  {document_id}: {count}")

    print("\nSample chunk:")
    print(chunks[0])