import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr

# ── All globals start as None ────────────────────────────────
_openai_client = None
_weaviate_client = None
_llm           = None

def get_clients():
    global _openai_client, _weaviate_client, _llm 

    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if _weaviate_client is None:
        import weaviate
        _weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=weaviate.auth.AuthApiKey(
                api_key=os.environ["WEAVIATE_API_KEY"]
            )
        )

    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        )

    return _openai_client, _weaviate_client, _llm


# ── LIFESPAN: WARM UP SERVICES BEFORE ACCEPTING TRAFFIC ───────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs BEFORE the server starts listening for requests
    print("[Startup] Initializing API Clients & Warming up Reranker...")
    
    # 1. Initialize API Clients
    get_clients()
    
    # 2. Warm up the CrossEncoder model from chat.py
    try:
        from src.chat import preload_reranker
        preload_reranker()
    except ImportError:
        try:
            from src.chat import preload_reranker
            preload_reranker()
        except Exception as e:
            print(f"Could not preload reranker: {e}")
            
    print("[Startup] All models loaded. System ready to receive traffic.")
    yield
    # This runs when the server shuts down
    print("Shutting down backend.")


# Initialize FastAPI with the lifespan manager
app = FastAPI(title="AEGIS Policy Assistant API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class ChunkInfo(BaseModel):
    document_id: str
    section: str
    score: float
    text_preview: str

class TokenInfo(BaseModel):
    total_tokens_before: int
    total_tokens_after: int
    chunks_before: int
    chunks_after: int
    truncated: bool
    dropped_chunks: int
    budget: int

class QueryResponse(BaseModel):
    answer: str
    category_detected: str | None
    sources: list[ChunkInfo]
    concepts_used: list[str]
    token_info: TokenInfo | None = None


# ── Routes ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "gpt-4o-mini"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    from src.chat import ask

    openai_client, weaviate_client, llm = get_clients()

    result = ask(
        query=request.query,
        session_id=request.session_id,
        weaviate=weaviate_client,
        openai_client=openai_client,
        llm=llm,
    )

    sources = []
    for chunk in result["sources"]:
        sources.append(ChunkInfo(
            document_id=chunk.get("document_id", "Unknown"),
            section=chunk.get("section", ""),
            score=round(float(chunk.get("score", 0)), 4),
            text_preview=chunk.get("text_preview", "")[:400],
        ))

    concepts = [
        "Multi-Query Expansion", "HyDE", "Metadata Pre-Filter",
        "RRF Fusion", "Post-Filter by Date", "Cross-Encoder Reranking",
    ]

    token_info = None
    if result.get("token_info"):
        ti = result["token_info"]
        token_info = TokenInfo(
            total_tokens_before=ti.get("total_tokens_before", 0),
            total_tokens_after=ti.get("total_tokens_after", 0),
            chunks_before=ti.get("chunks_before", 0),
            chunks_after=ti.get("chunks_after", 0),
            truncated=ti.get("truncated", False),
            dropped_chunks=ti.get("dropped_chunks", 0),
            budget=ti.get("budget", 3000),
        )

    return QueryResponse(
        answer=result["answer"],
        category_detected=result.get("category_detected"),
        sources=sources,
        concepts_used=concepts,
        token_info=token_info,
    )