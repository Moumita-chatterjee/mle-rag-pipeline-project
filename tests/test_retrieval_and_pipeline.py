from pathlib import Path

import pytest

from src.embedding import load_model
from src.ingestion import load_chunks
from src.pipeline import answer_question
from src.retrieval import build_index, open_client, retrieve

@pytest.fixture(scope="module")
def index():
    model = load_model()
    client = open_client(None)
    build_index(client, model, load_chunks(Path("data/knowledge_base")))
    yield client, model
    client.close()

def test_supported_question_retrieves_the_right_document(index):
    client, model = index
    result = retrieve(client, model, "How long is a password reset link valid?")

    assert not result.abstain
    assert result.evidence[0][1].document_id == "account-password"

def test_unsupported_question_abstains(index):
    client, model = index
    result = retrieve(client, model, "Is there a discounted subscription rate for students?")

    assert result.abstain

def test_pipeline_answers_with_citations(index):
    client, model = index
    answer = answer_question(client, model, "How long is a password reset link valid?")

    assert answer.mode == "extractive"
    assert not answer.abstained
    assert "[account-password]" in answer.text
    assert answer.citation_problems == []

def test_pipeline_removes_injected_instructions(index):
    client, model = index
    answer = answer_question(
        client, model, "What is the example of malicious content used for security testing?"
    )

    assert "retrieved-content-safety:1" in answer.flagged
    assert "IGNORE ALL PRIOR INSTRUCTIONS" not in answer.text

