from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownTextSplitter, MarkdownHeaderTextSplitter
import os
from pprint import pprint

from ingestion.file_chunker import retrieve_chunks 
from ingestion.metadata_tagger import metadata_tagging


chunk_size = 500
chunk_overlap = int(chunk_size*0.13)    


def process_files(src_files: str):
    for root, dirs, files in os.walk(src_files):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                #print(f"Reading file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    chunks = retrieve_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    metadata = metadata_tagging(chunks)
                    pprint(metadata)
                    #pprint(chunks)

if __name__ == "__main__":
    process_files("./data")