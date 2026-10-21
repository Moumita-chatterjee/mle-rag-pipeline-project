
from pathlib import Path

def list_documents(folder:Path) -> list[Path]:
    list_paths = sorted(folder.glob("*.md"))
    return list_paths




if __name__=="__main__":
    knowledge_folder = Path("data/knowledge_base")
    paths = list_documents(knowledge_folder)

    print(f"found {len(paths)} documents")
    for path in paths:
        #.stem is a file name without extension
        print(path.stem)
