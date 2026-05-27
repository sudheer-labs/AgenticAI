from pprint import pprint
import os
import json
from src.ingestion.embedding import generate_batch_embedding, search_collection
from collections import defaultdict


def generate_multi_query(user_query: str, llm, query_count: int = 3):

    prompt = f"""
    You are an Enterprise RAG Query Expansion Engine for a corporate policy document retrieval system.

    Your task is to generate {query_count} semantically diverse search queries from a single user query. 
    The generated queries will be used to improve retrieval quality from a vector database and keyword-based enterprise search system.

    Guidelines:
    - Preserve the original user intent.
    - Generate concise, enterprise-style search queries.
    - Expand abbreviations where relevant.
    - Optimize for document retrieval, not conversational responses.
    - Queries should be diverse but highly relevant.
    - Avoid duplicate or overly similar queries.
    - Do NOT explain the queries.
    - Do NOT number the queries.
    - Return ONLY a valid Python list of strings.
    - The output must be directly parseable.

    Example:
    User Query:
    "Can I expense a taxi?"

    Expected Output:
    [
        "Taxi reimbursement policy",
        "Ground transportation expense eligibility",
        "Corporate travel cab reimbursement"
    ]

    User Query:
    {user_query}

    ai_response:
    """

    try:
        response = llm.invoke(prompt)

        raw_text = response.content.strip()

        queries = json.loads(raw_text)

        # Validate response
        if (
            not isinstance(queries, list)
            or len(queries) == 0
        ):
            return [user_query]

        # Keep only valid non-empty strings
        queries = [
            q.strip()
            for q in queries
            if isinstance(q, str) and q.strip()
        ]

        if len(queries) == 0:
            return [user_query]
        
        # Adding oringal user query
        final_queries = [user_query]

        # Adding unique queries from the response while preserving order
        for q in queries:
            if q.lower() != user_query.lower():
                final_queries.append(q)

        # Remove duplicates while preserving order
        final_queries = list(dict.fromkeys(final_queries))

        return final_queries[: query_count + 1]

    except Exception as e:
        print(f"Multi-query generation failed: {e}")

        # Fallback
        return [user_query]

def multi_query_search(queries_list: list, wv_client, top_k: int = 5):

    all_ranked_results = []
    for query in queries_list:
        results = search_collection(query, wv_client, top_k = top_k)
        #pprint(f"Search Results for '{query}': {results}")
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
            result["query"] = query
        all_ranked_results.extend(results)
    # RRF algorithm is used in information retrieval and AI search systems (like RAG and hybrid search) to combine multiple ranked lists of results from different sources into one, single optimized list
    # It improves search quality by combining different methods (e.g., combining keyword search and semantic vector search) to achieve better accuracy than either method alone.
    
    doc_store = {}
    rrf_scores = defaultdict(float)

    # Standard RRF constant
    k = 60

    for item in all_ranked_results:
        pprint(f"Processing item: {item}")
        # Unique document identifier
        doc_id = item["id"]
        rank = item["rank"]

        # Store latest/full document
        doc_store[doc_id] = item

        # RRF formula
        rrf_scores[doc_id] += 1 / (k + rank)

    # Sort documents by fused RRF score
    fused_doc_ids = sorted(
        rrf_scores.keys(),
        key=lambda x: rrf_scores[x],
        reverse=True
    )

    # Build final response
    final_results = []

    for doc_id in fused_doc_ids[:top_k]:

        doc = doc_store[doc_id]

        # Optional: expose RRF score
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)

        final_results.append(doc)

    return final_results

# if __name__ == "__main__":
#     from langchain_openai import ChatOpenAI
#     pprint("Testing multi-query generation and search...")
#     llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)
#     user_query = "What is the policy for remote work and flexible hours?"
#     queries = generate_multi_query(user_query, llm)
#     pprint(queries)
#     multi_query_search(queries, top_k=5)