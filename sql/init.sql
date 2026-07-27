-- Enable the pgvector extension (adds the `vector` type + distance operators)
CREATE EXTENSION IF NOT EXISTS vector;

-- Core table: every retrievable chunk lives here, alongside its embedding
-- AND a full-text search vector, so hybrid search is a single-table query.
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    doc_id          TEXT NOT NULL,          -- e.g. "policy_HP-2024-119"
    doc_title       TEXT NOT NULL,          -- human-readable source name
    section         TEXT,                   -- e.g. "Section 4.2 - Maternity Waiting Period"
    policy_type     TEXT,                   -- e.g. "health", "motor" -- used for metadata filtering
    content         TEXT NOT NULL,          -- the raw chunk text
    embedding       vector(384),            -- 384 dims matches all-MiniLM-L6-v2; change if you swap models
    content_tsv     tsvector,               -- full-text index for BM25-style keyword search
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Keep content_tsv in sync automatically whenever content changes
CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks;
CREATE TRIGGER trg_chunks_tsv
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update();

-- Vector index (HNSW) for fast approximate nearest-neighbor search.
-- cosine distance operator class since that's the standard for text embeddings.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- GIN index for fast full-text (BM25-style) search
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
    ON chunks USING gin (content_tsv);

-- Optional metadata filter index
CREATE INDEX IF NOT EXISTS chunks_policy_type_idx ON chunks (policy_type);


-- A minimal mock "claims" table so the agent (Phase 2) has something
-- resembling a real backend system to call as a tool.
CREATE TABLE IF NOT EXISTS claims (
    claim_id        TEXT PRIMARY KEY,
    policy_id       TEXT NOT NULL,
    claim_amount    NUMERIC NOT NULL,
    status          TEXT NOT NULL,          -- e.g. "under_review", "approved", "denied"
    filed_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO claims (claim_id, policy_id, claim_amount, status) VALUES
    ('CLM-2024-8817', 'HP-2024-119', 45000, 'under_review'),
    ('CLM-2024-9021', 'HP-2024-119', 850000, 'pending_human_approval')
ON CONFLICT (claim_id) DO NOTHING;
