from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownTextSplitter, MarkdownHeaderTextSplitter
import os
from pprint import pprint

from src.ingestion.file_chunker import retrieve_chunks 
from src.ingestion.metadata_tagger import metadata_tagging
from src.ingestion.embedding import upsert_chunks_batch
from langchain_openai import ChatOpenAI
from src.retrieve.retrieval import generate_multi_query, multi_query_search
import weaviate
from weaviate.classes.init import Auth

chunk_size = 500
chunk_overlap = int(chunk_size*0.13)    


llm = ChatOpenAI(model="gpt-4o-mini", 
                 api_key=os.getenv("OPENAI_API_KEY")
                 , temperature=0.2)

wv_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["WEAVIATE_URL"],
    auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
    )

def process_files(src_files: str):
    for root, dirs, files in os.walk(src_files):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                #print(f"Reading file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    chunks = retrieve_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    metadata_chunks = metadata_tagging(chunks, llm)
                    #pprint(metadata[:5])
                    upsert_chunks_batch(metadata_chunks, wv_client, 50)
                    print("Ingestion Completed for file: ", file_path)

if __name__ == "__main__":

    process_files("./data")
    # user_query = "What is the policy for remote work and flexible hours?"0
    # queries = generate_multi_query(user_query, llm)
    # pprint(queries)
    # res = multi_query_search(queries, top_k=5)
    # pprint(f"MQ Search Results: {res}")