from sentence_transformers import SentenceTransformer
from embedding import encode_text


def split_text_to_chunks_tokens(
    data: list,
    embedder: SentenceTransformer,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list:
    """
    Token-based chunking. Mirror of CairoS1 split_text_to_chunks_tokens().

    Input : list of dicts from extract_engine_events()
            each has {unit_id, cycle_id, rul, failure_mode, zone, text}
    Output: list of chunk dicts (no embeddings yet)
    """
    tokenizer = embedder.tokenizer
    chunks    = []

    for item in data:
        text         = item.get("text", "")
        unit_id      = item.get("unit_id")
        cycle_id     = item.get("cycle_id")
        rul          = item.get("rul")
        failure_mode = item.get("failure_mode")
        zone         = item.get("zone")

        if not text.strip():
            continue

        tokens        = tokenizer.encode(text)
        tokens_length = len(tokens)

        if tokens_length == 0:
            continue

        start = 0
        while start < tokens_length:
            end          = min(start + chunk_size, tokens_length)
            chunk_tokens = tokens[start:end]
            chunk_text   = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            if chunk_text.strip():
                chunks.append({
                    "unit_id":      unit_id,
                    "cycle_id":     cycle_id,
                    "rul":          rul,
                    "failure_mode": failure_mode,
                    "zone":         zone,
                    "chunk_text":   chunk_text.strip(),
                })

            # Move forward — stop if we've reached the end
            next_start = start + chunk_size - overlap
            if next_start <= start:          # safety: prevent infinite loop
                next_start = start + 1
            start = next_start

    return chunks


def add_embeddings_to_chunks(
    chunks: list,
    embedder: SentenceTransformer,
) -> list:
    """
    Add embedding to each chunk. Mirror of CairoS1 add_embeddings_to_chunks().
    Skips empty chunks.
    """
    enriched = []

    for chunk in chunks:
        text = chunk.get("chunk_text", "").strip()
        if not text:
            continue

        embedding = encode_text(embedder, text)
        enriched.append({**chunk, "embedding": embedding})

    return enriched
