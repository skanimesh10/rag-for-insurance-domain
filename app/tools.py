"""
A minimal mock 'claims API' -- stands in for a real external backend
system the agent needs to call as a tool. In production this would be
an HTTP call to another service; here it's a query against the
`claims` table seeded in sql/init.sql, which is enough to demonstrate
the tool-calling pattern without standing up a second service.
"""
import re
import asyncpg

CLAIM_ID_RE = re.compile(r"CLM-\d{4}-\d+", re.IGNORECASE)


def extract_claim_id(text: str) -> str | None:
    """Pull a claim ID like 'CLM-2024-8817' out of free text, if present."""
    match = CLAIM_ID_RE.search(text)
    return match.group(0).upper() if match else None


async def get_claim_status(pool: asyncpg.Pool, claim_id: str) -> dict | None:
    """The 'tool call' itself -- looks up a claim by ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT claim_id, policy_id, claim_amount, status FROM claims WHERE claim_id = $1",
            claim_id,
        )
    return dict(row) if row else None
