from src.ingestion.embedding import generate_batch_embedding, search_collection
from weaviate.classes.query import Filter

def generate_hypothetical_answer(user_query: str, llm):
    
    prompt = f"""
You're a Corporta Polciy Assistant.
Your role is to generate the hypothetical answer for the user query.
Guidelines:
- Generate concise, enterprise-style answers not more than 100 words.
- Answer should be based on the information that would typically be found in corporate policy documents.
- Avoid speculative or overly broad answers like 'I don't know' or 'I'm not sure' or 'hypothetical' etc.

Question: {user_query}

Answer:
"""
    return llm.invoke(prompt).content.strip()

def hypothetical_answer_search(user_query: str, wv_client, llm, top_k=5, category: str | None = None):

    hypothetical_answer = generate_hypothetical_answer(user_query, llm)

    print(f"Hypothetical Answer: {hypothetical_answer}")

    qry_filter = None

    if category is not None:
        qry_filter = Filter.by_property("policy_category").equal(category)

    # PASS TEXT, NOT VECTOR
    results = search_collection(
        hypothetical_answer,
        wv_client,
        top_k=top_k,
        filters=qry_filter
    )

    response = []

    for obj in results:

        response.append({
            "id": obj.get("id"),
            "score": 1 - obj.get("distance", 0),
            "text": obj.get("content", ""),
            "metadata": obj.get("metadata", {})
        })

    return response
