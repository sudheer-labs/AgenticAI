# retrieval/reranker.py
# Uses Cohere Rerank API on deployment (zero RAM, free tier = 1000 calls/month)
# Falls back to CrossEncoder locally when COHERE_API_KEY is not set

import os
import numpy as np

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

_rerank_model   = None
_cohere_client  = None

COHERE_API_KEY  = os.getenv("COHERE_API_KEY")


def get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
        print("==> Cohere Rerank client loaded.", flush=True)
    return _cohere_client


def get_reranker():
    """Load CrossEncoder locally (when no COHERE_API_KEY set)."""
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            model_kwargs={"cache_dir": "./models/cache"}
        )
        print("==> CrossEncoder loaded.", flush=True)
    return _rerank_model


def _cohere_rerank(query: str, chunks: list, top_k: int) -> list:
    """Rerank using Cohere Rerank API — zero RAM, works on Render free tier."""
    client = get_cohere_client()

    texts = [chunk["text"][:512] for chunk in chunks]  # Cohere max ~512 chars per doc

    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_k,
    )

    # Build reranked list in order of Cohere's ranking
    reranked = []
    for i, result in enumerate(response.results):
        chunk = chunks[result.index]
        # Normalize relevance score to 0-1
        chunk["rerank_score"] = round(float(result.relevance_score), 4)
        reranked.append(chunk)

    return reranked


def _crossencoder_rerank(query: str, chunks: list, top_k: int) -> list:
    """Rerank using local CrossEncoder — used in local development."""
    model = get_reranker()
    pairs = [[query, chunk["text"]] for chunk in chunks]
    raw_scores = model.predict(pairs)

    if len(raw_scores) > 1:
        mn, mx = raw_scores.min(), raw_scores.max()
        scores = (raw_scores - mn) / (mx - mn) if mx > mn else np.ones_like(raw_scores)
    else:
        scores = np.array([1.0])

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    if COHERE_API_KEY:
        print("==> Using Cohere Rerank API.", flush=True)
        return _cohere_rerank(query, chunks, top_k)
    else:
        print("==> Using CrossEncoder (local).", flush=True)
        return _crossencoder_rerank(query, chunks, top_k)
