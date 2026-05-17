# chat.py
from unicodedata import category

from dotenv import load_dotenv
from langgraph_sdk import Auth
load_dotenv()
from collections import defaultdict
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

from src.retrieve.filtering import detect_category, prefilter_search, post_filter_by_date
from src.retrieve.retrieval import generate_multi_query
from src.retrieve.reranking import rerank, get_reranker
from src.retrieve.hyde import hypothetical_answer_search
from src.utils.token_budget import enforce_token_budget

session_store = {}


def preload_reranker():
    """Called at startup to warm up the CrossEncoder model.
    Prevents Render's 30s request timeout from killing the first /ask call.
    """
    import logging
    logger = logging.getLogger("uvicorn.error")
    logger.info("Warming up CrossEncoder model...")
    get_reranker()
    logger.info("CrossEncoder ready.")


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def build_context(chunks: list) -> str:

    parts = []

    for i, chunk in enumerate(chunks, start=1):

        metadata = chunk.get("metadata") or {}

        source = metadata.get("document_id", "Unknown")
        category = metadata.get("policy_category", "")
        text = chunk.get("text", "")

        parts.append(
            f"[Source {i} - {source} ({category})]:\n{text}"
        )

    return "\n\n".join(parts)


def ask(query: str, session_id: str, weaviate, openai_client: OpenAI, llm) -> dict:

    # Step 1: detect category
    category = detect_category(query, llm)

    # Step 2: generate multiple queries
    queries = generate_multi_query(query, llm)
    print(f"Queries generated: {queries}")
    # Step 3: search with pre-filter for each query variant
    all_ranked_lists = []
    for q in queries:
        results = prefilter_search(q, llm, weaviate, top_k=13)
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        all_ranked_lists.append(results)

    # Step 3b: HyDE search
    hyde_results = hypothetical_answer_search(query, weaviate, llm, top_k=13, category=category)
    for rank, item in enumerate(hyde_results, start=1):
        item["rank"] = rank
    all_ranked_lists.append(hyde_results)

    # Step 4: RRF fusion
    doc_store = {}
    rrf_scores = defaultdict(float)
    for ranked_list in all_ranked_lists:
        for item in ranked_list:
            doc_id = item["id"]
            rank = item["rank"]
            doc_store[doc_id] = item
            rrf_scores[doc_id] += 1 / (60 + rank)
    fused = sorted(doc_store.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused_chunks = [doc_store[doc_id] for doc_id in fused[:25]]

    # Step 5: post filter by date
    filtered_chunks = post_filter_by_date(fused_chunks)

    # Step 6: rerank → top 5
    final_chunks = rerank(query, filtered_chunks, top_k=5)

    # Step 7: enforce token budget
    final_chunks, token_info = enforce_token_budget(final_chunks, budget=3000)
    context = build_context(final_chunks)

    # Step 8: build prompt with chat history
    history = get_session_history(session_id)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a corporate policy assistant.
Answer questions using ONLY the context below.
Be clear and helpful.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])

    # Step 9: get answer from LLM
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "history": history.messages,
        "query": query
    })

    # Step 10: save to history
    history.add_user_message(query)
    history.add_ai_message(response.content)

    # Step 11: build sources list
    raw_scores = [float(chunk.get("rerank_score", 0)) for chunk in final_chunks]
    min_s = min(raw_scores) if raw_scores else 0
    max_s = max(raw_scores) if raw_scores else 1
    score_range = max_s - min_s if max_s != min_s else 1

    sources = []

    for chunk in final_chunks:
        metadata = chunk.get("metadata") or {}
        sources.append({
            "document_id": metadata.get("document_id", "Unknown"),
            "score": round(
                (float(chunk.get("rerank_score", 0)) - min_s)
                / score_range,
                4
            ),

            "text_preview": chunk.get("text", "")[:300],
            "section": metadata.get("h2_header", ""),
            "policy_category": metadata.get("policy_category", ""),
            "effective_date": metadata.get("effective_date", ""),
        })
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"Final answer: {response.content}")
    print(f"Sources used: {sources}")
    return {
        "answer": response.content,
        "sources": sources,
        "category_detected": category,
        "token_info": token_info,
    }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from pydantic import SecretStr

    load_dotenv()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    wv_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["WEAVIATE_URL"],
    auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"])
    )

    result = ask(
        query="What is the Travel expense claim policy?",
        session_id="test_user_1",
        weaviate=wv_client,
        openai_client=openai_client,
        llm=llm
    )
    wv_client.close()
    # print(f"Category detected: {result['category_detected']}")
    # print(f"\nAnswer:\n{result['answer']}")
    # print(f"\nSources used:")
    # for s in result["sources"]:
    #     print(f"  - {s['document_id']} | {s['section']}")
