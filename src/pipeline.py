import sys
from dataclasses import dataclass

from openai import OpenAI                                   # NEW
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.embedding import load_model
from src.generation import (                                # CHANGED: one import, not two
    ABSTAIN_MESSAGE,
    find_citation_problems,
    generate_answer,
    generate_with_llm,
    load_llm,
)
from src.ingestion import Chunk
from src.retrieval import open_client, retrieve
from src.safety import filter_evidence


@dataclass(frozen=True)
class Answer:
    question: str
    mode: str                                               # NEW
    text: str
    abstained: bool
    top_score: float
    retrieved: list[tuple[float, Chunk]]
    flagged: list[str]
    citation_problems: list[str]


def answer_question(
    client: QdrantClient,
    model: SentenceTransformer,
    question: str,
    llm: tuple[OpenAI, str] | None = None,                  # NEW
) -> Answer:
    """Retrieve, filter unsafe chunks, generate a cited answer, and validate it."""
    result = retrieve(client, model, question)
    safe, flagged = filter_evidence(result.evidence)
    mode = "llm" if llm else "extractive"                   # NEW

    text = ABSTAIN_MESSAGE
    problems = []
    if not result.abstain and safe:
        if llm:                                             # NEW
            text = generate_with_llm(llm[0], llm[1], question, safe)
        else:
            text = generate_answer(safe)
        problems = find_citation_problems(text, safe)
        if problems:
            text = ABSTAIN_MESSAGE

    return Answer(
        question=question,
        mode=mode,                                          # NEW
        text=text,
        abstained=text == ABSTAIN_MESSAGE,
        top_score=result.top_score,
        retrieved=result.evidence,
        flagged=flagged,
        citation_problems=problems,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "How do I get a refund for a duplicate payment?"
    client = open_client()
    try:
        answer = answer_question(client, load_model(), question, load_llm())   # CHANGED
        print(f"Mode:      {answer.mode}")                  # NEW
        print(f"Question:  {answer.question}")
        print(f"Top score: {answer.top_score:.3f}")
        print(f"Retrieved: {[chunk.chunk_id for _, chunk in answer.retrieved]}")
        print(f"Flagged:   {answer.flagged}")
        print(f"Citation problems: {answer.citation_problems}")
        print()
        print(answer.text)
    finally:
        client.close()