from sentence_transformers import SentenceTransformer


def load_embedder(MODEL_NAME):
    model = SentenceTransformer(MODEL_NAME)
    return model


def encode_text(embedder_model, text):
    embedding = embedder_model.encode(text, normalize_embeddings=True)
    return embedding
