import os
import uuid
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from pprint import pprint
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

#filters = Filter.by_property("department").equal("HR")

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

CLASS_NAME = "AegisPolicy"

def create_weaviate_schema(wv_client):

    existing_collections = wv_client.collections.list_all()

    if CLASS_NAME not in existing_collections:

        wv_client.collections.create(
            name=CLASS_NAME,

            vectorizer_config=Configure.Vectorizer.none(),

            properties=[

                Property(
                    name="content",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="header1",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="header2",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="header3",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="document_id",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="effective_date",
                    data_type=DataType.DATE
                ),

                Property(
                    name="policy_category",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="policy_owner",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="has_table",
                    data_type=DataType.BOOL
                ),

                Property(
                    name="chunk_id",
                    data_type=DataType.TEXT
                )
            ]
        )

        print(f"Created collection: {CLASS_NAME}")

    else:
        print(f"Collection already exists: {CLASS_NAME}")

def generate_batch_embedding(batch_texts: list):

    if isinstance(batch_texts, str):
        return embedding_model.embed_query(batch_texts)

    if isinstance(batch_texts, list):
        if len(batch_texts) == 1:
            return embedding_model.embed_query(batch_texts[0])
        else:
            return embedding_model.embed_documents(batch_texts)

def convert_to_rfc3339(date_str):

    if not date_str:
        return None
    try:

        dt = datetime.strptime(
            date_str,
            "%Y-%m-%d"
        )

        return dt.isoformat() + "Z"

    except Exception:
        return None
    
def upsert_chunks_batch(chunks: list, wv_client, batch_size: int = 50):

    create_weaviate_schema(wv_client)
    collection = wv_client.collections.get(CLASS_NAME)

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_texts = [chunk["content"] for chunk in batch_chunks]
        batch_embeddings = generate_batch_embedding(batch_texts)
        with collection.batch.dynamic() as batch:
            batch.batch_size = batch_size

            for chunk, vector in zip(batch_chunks, batch_embeddings):
                
                chunk_id = str(uuid.uuid4())
                metadata = chunk.get("metadata", {})
                metadata["effective_date"] = convert_to_rfc3339(
                    metadata.get("effective_date")
                )
                properties = {
                    "content": chunk.get("content", ""),
                    **metadata,
                    "has_table": chunk.get("has_table", False),
                    "chunk_id": chunk_id
                }
                
                batch.add_object(
                    properties=properties,
                    uuid=chunk_id,
                    vector=vector
                )
        print(f"Inserted batch {(i // batch_size) + 1}")
    #wv_client.close()

def delete_collection(wv_client):

    existing_collections = wv_client.collections.list_all()

    if CLASS_NAME in existing_collections:

        wv_client.collections.delete(CLASS_NAME)

        print(f"Deleted collection: {CLASS_NAME}")

    else:
        print(f"Collection does not exist: {CLASS_NAME}")

def search_collection(query: str, wv_client, top_k: int = 5, filters: Filter = None):

    try:

        collection = wv_client.collections.get(CLASS_NAME)

        query_vector = generate_batch_embedding(query)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=filters if filters else None,
            return_metadata=["distance"]
        )

        print(f"Vector Search - Query: {query}, Results found: {len(response.objects)}")
        results = []

        for obj in response.objects:

            results.append({
                "id": str(obj.uuid),
                "content": obj.properties.get("content"),
                "metadata": obj.properties,
                "distance": obj.metadata.distance
            })

        return results

    except Exception as e:

        print(f"Vector search failed: {e}")

        return []
# if __name__ == "__main__":
#     delete_collection(wv_client)