import psycopg
def get_connection(PG_CONN_STR: str):
    """Open psycopg connection."""
    conn = psycopg.connect(PG_CONN_STR)
    return conn


def create_chunks_table(conn, table_name: str, dim: int):
    """
    Create RAG table if not exists. Run once.
    PostgreSQL must have pgvector:
        CREATE EXTENSION IF NOT EXISTS vector;
    """
    with conn.cursor() as cur:
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id           SERIAL PRIMARY KEY,
            unit_id      INT,
            cycle_id     INT,
            rul          FLOAT,
            failure_mode TEXT,
            zone         TEXT,
            chunk_text   TEXT,
            embedding    VECTOR({dim})
        );
        """)
        conn.commit()
    print(f"[db] Table '{table_name}' ready (dim={dim}).")


def drop_table(conn, table_name: str):
    """Drop table — use to reset during development."""
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
    print(f"[db] Table '{table_name}' dropped.")


def insert_chunks_batch(conn, table_name: str, chunks: list):
    """
    Batch insert enriched chunks into PostgreSQL.
    Mirror of CairoS1 insert_chunks_batch().
    """
    records = []
    for c in chunks:
        emb = c.get("embedding")
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        records.append((
            c.get("unit_id"),
            c.get("cycle_id"),
            c.get("rul"),
            c.get("failure_mode"),
            c.get("zone"),
            c.get("chunk_text"),
            emb,
        ))

    with conn.cursor() as cur:
        cur.executemany(f"""
        INSERT INTO {table_name}
            (unit_id, cycle_id, rul, failure_mode, zone, chunk_text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, records)

    conn.commit()
    print(f"[db] Inserted {len(records)} chunks into '{table_name}'.")


def retrieve_top_k(
    conn,
    table_name: str,
    query_embedding,
    k: int = 5,
    zone_filter: str = None,
    failure_mode_filter: str = None,
) -> list:
    """
    Cosine similarity search with optional filters.
    Mirror of CairoS1 retrieve_top_k().

    BUG FIX: SQL is built with WHERE conditions BEFORE the
    ORDER BY embedding clause — params must match this order exactly.

    zone_filter         : 'healthy' | 'degrading' | 'critical (alert zone)'
    failure_mode_filter : 'HPT_efficiency_degradation' | 'LPT_efficiency_flow_HPT_combined'
    """
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    # Build WHERE clause
    conditions    = []
    filter_params = []

    if zone_filter:
        conditions.append("zone = %s")
        filter_params.append(zone_filter)
    if failure_mode_filter:
        conditions.append("failure_mode = %s")
        filter_params.append(failure_mode_filter)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    #param order must match SQL exactly:
    # 1. filter params (for WHERE clause)
    # 2. query_embedding (for cosine_sim SELECT)
    # 3. query_embedding (for ORDER BY)
    # 4. k (for LIMIT)
    sql = f"""
    SELECT
        unit_id,
        cycle_id,
        rul,
        failure_mode,
        zone,
        chunk_text,
        1 - (embedding <=> %s::vector) AS cosine_sim
    FROM {table_name}
    {where}
    ORDER BY embedding <=> %s::vector
    LIMIT %s
    """

    params = [query_embedding] + filter_params + [query_embedding, k]

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        {
            "unit_id":      row[0],
            "cycle_id":     row[1],
            "rul":          row[2],
            "failure_mode": row[3],
            "zone":         row[4],
            "chunk_text":   row[5],
            "score":        round(float(row[6]), 4),
        }
        for row in rows
    ]
