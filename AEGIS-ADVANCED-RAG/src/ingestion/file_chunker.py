from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
from pprint import pprint


headers_to_split_on = [
                    ("#", "header1"),
                    ("##", "header2"),
                    ("###", "header3"),
                ]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,
                                               strip_headers=False)

def retrieve_chunks(text: str, chunk_size: int, chunk_overlap: int):
    
    header_chunks = markdown_splitter.split_text(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",   # Paragraphs
            "\n",     # Lines
            " ",      # Words
            ""]       # Characters
    )
    final_chunks = []
    for doc in header_chunks:
        content = doc.page_content
        metadata = doc.metadata

        # Skip Table of Contents chunks
        if metadata.get("header2", "").lower() == "table of contents":
            print("Skipping Table of Contents chunk")
            continue
        lines = content.strip().split("\n")
        table_lines = [l for l in lines if l.strip().startswith("|")]

        if table_lines:
            table_text = "\n".join(table_lines)

            # Chunk large tables if the table items are more than 3 rows (excluding header and separator)
            if len(table_lines) > 5:
                row_chunks = split_large_table(table_text)
                for row_chunk in row_chunks:
                    final_chunks.append({
                        "content": row_chunk,
                        "metadata": metadata,
                        "has_table": True
                    })
            else:
                # Preserve small tables as-is to maintain readability
                final_chunks.append({
                    "content": content,
                    "metadata": metadata,
                    "has_table": True
                })
        else:
            sub_chunks = text_splitter.split_text(content)
            for chunk in sub_chunks:
                final_chunks.append({
                    "content": chunk,
                    "metadata": metadata,
                    "has_table": False
                })

    return final_chunks

def split_large_table(table_text: str):
    lines = table_text.strip().split("\n")
    header_line = lines[0]
    separator_line = lines[1]
    data_lines = lines[2:]

    row_chunks = []
    for line in data_lines:
        chunk = f"{header_line}\n{separator_line}\n{line}"
        row_chunks.append(chunk)
    return row_chunks

# table = """| Destination | Daily Allowance |
# |-------------|-----------------|
# | USA         | $80             |
# | Europe      | $70             |
# | Asia        | $60             |"""

# if __name__ == "__main__":
#     result = split_large_table(table)
#     for i, chunk in enumerate(result):
#         print(f"--- Table Chunk {i+1} ---")
#         print(chunk)
#         print()