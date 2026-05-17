"""
utils/token_budget.py
─────────────────────
Token budget enforcer for Aegis retrieval pipeline.

Counts tokens across reranked chunks before the LLM call.
Trims from lowest-ranked chunk upward if total exceeds budget.
Logs a warning when truncation occurs.

Usage (in chat.py):
    from utils.token_budget import enforce_token_budget
    chunks, token_info = enforce_token_budget(reranked_chunks, budget=3000)
    # pass token_info to Streamlit sidebar for display
"""

import logging
import tiktoken

logger = logging.getLogger(__name__)

# Model used in Aegis — gpt-4o-mini uses cl100k_base encoding
ENCODING_NAME = "cl100k_base"

# Safe token budget for context chunks (leaves headroom for system
# prompt + user query + LLM response within gpt-4o-mini's 128k limit)
DEFAULT_CHUNK_BUDGET = 3000


def count_tokens(text: str, encoding_name: str = ENCODING_NAME) -> int:
    """Return the number of tokens in a string."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def enforce_token_budget(
    chunks: list[dict],
    budget: int = DEFAULT_CHUNK_BUDGET,
    encoding_name: str = ENCODING_NAME,
) -> tuple[list[dict], dict]:
    """
    Enforce a token budget across a list of reranked chunks.

    Chunks are expected to be ordered best-first (CrossEncoder top-1 first).
    Trimming removes from the END (lowest-ranked) first, preserving the
    highest-quality context.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have a 'text' key with the chunk content.
        May also have 'score', 'document_id', 'section' etc.
    budget : int
        Maximum total tokens allowed across all chunks.
    encoding_name : str
        tiktoken encoding to use.

    Returns
    -------
    chunks_kept : list[dict]
        Subset of input chunks that fit within budget.
    token_info : dict
        Metadata for logging / Streamlit sidebar display:
        {
            "total_tokens_before": int,
            "total_tokens_after": int,
            "chunks_before": int,
            "chunks_after": int,
            "truncated": bool,
            "dropped_chunks": int,
            "budget": int,
        }
    """
    if not chunks:
        return [], {
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "chunks_before": 0,
            "chunks_after": 0,
            "truncated": False,
            "dropped_chunks": 0,
            "budget": budget,
        }

    # Count tokens per chunk once (avoid re-encoding)
    chunk_tokens = [count_tokens(c["text"], encoding_name) for c in chunks]
    total_before = sum(chunk_tokens)

    # Greedily keep chunks from the top (best-ranked) until budget exhausted
    kept = []
    running_total = 0
    for chunk, tok_count in zip(chunks, chunk_tokens):
        if running_total + tok_count <= budget:
            kept.append(chunk)
            running_total += tok_count
        else:
            # Stop — remaining chunks (lower-ranked) are dropped
            break

    total_after = running_total
    truncated = len(kept) < len(chunks)
    dropped = len(chunks) - len(kept)

    if truncated:
        logger.warning(
            "Token budget enforced: %d chunks → %d chunks kept "
            "(%d tokens → %d tokens, budget=%d, dropped=%d chunk(s))",
            len(chunks), len(kept),
            total_before, total_after,
            budget, dropped,
        )

    token_info = {
        "total_tokens_before": total_before,
        "total_tokens_after": total_after,
        "chunks_before": len(chunks),
        "chunks_after": len(kept),
        "truncated": truncated,
        "dropped_chunks": dropped,
        "budget": budget,
    }

    return kept, token_info
