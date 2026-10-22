from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_model(model_name: str = MODEL_NAME ) -> SentenceTransformer:
    return SentenceTransformer(model_name, device = "cpu")


if __name__ =="__main__":
    model = load_model()
    sentences = [
        "How do I get a refund for a duplicate payment?",
        "Northstar refunds duplicate payments when two transactions match one invoice.",
        "Clients receiving 429 should retry using exponential backoff.",
    ]
    vectors = model.encode(sentences, normalize_embeddings = True)

    print("Shape:", vectors.shape)
    print("first 5 numbers of sentence 0:", vectors[0][:5])
    print("Ques vs refund policy: ", float(vectors[0] @ vectors[1]))
    print("Ques vs API Limits: ", float(vectors[0] @ vectors[2]))