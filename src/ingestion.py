
from pathlib import Path

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


if __name__=="__main__":
    knowledge_folder = Path("data/knowledge_base")
    paths = list_documents(knowledge_folder)

    print(f"found {len(paths)} documents")
    

    title, body = read_document(paths[2])
    paragraphs = split_paragraph(body)
    print("title:",title)
    print(f"{len(paragraphs)} paragraphs")
    for i, paragraph in enumerate(paragraphs):
        print(f"[{i}] ({len(paragraph)} chars) {paragraph}")
   
