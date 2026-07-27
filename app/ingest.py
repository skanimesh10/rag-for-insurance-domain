"""
One-off ingestion script. Run with: python -m app.ingest

Reads every .txt file in sample_docs/, chunks it (structural, by
Section header), embeds each chunk, and inserts into the `chunks`
table. content_tsv is filled automatically by the DB trigger.
"""
import asyncio
import re
from pathlib import Path

from app.db import get_pool, close_pool
from app.chunking import split_into_sections
from app.embeddings import embed_batch

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "sample_docs"

POLICY_TYPE_RE = re.compile(r"Policy Type:\s*(\w+)", re.IGNORECASE)


async def ingest_file(pool, path: Path) -> None:
    raw_text = path.read_text()
    doc_id = path.stem  # e.g. "policy_HP-2024-119"
    doc_title = raw_text.splitlines()[0].strip()

    policy_type_match = POLICY_TYPE_RE.search(raw_text)
    policy_type = policy_type_match.group(1).lower() if policy_type_match else None

    chunks = split_into_sections(raw_text)
    if not chunks:
        print(f"  no chunks produced for {path.name}, skipping")
        return

    embeddings = embed_batch([c.content for c in chunks])

    async with pool.acquire() as conn:
        for chunk, embedding in zip(chunks, embeddings):
            await conn.execute(
                """
                INSERT INTO chunks (doc_id, doc_title, section, policy_type, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                doc_id, doc_title, chunk.section, policy_type, chunk.content, embedding,
            )

    print(f"  ingested {len(chunks)} chunks from {path.name}")


async def main() -> None:
    pool = await get_pool()
    files = sorted(SAMPLE_DOCS_DIR.glob("*.txt"))
    print(f"Found {len(files)} document(s) to ingest")
    for path in files:
        print(f"Ingesting {path.name}...")
        await ingest_file(pool, path)
    await close_pool()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
