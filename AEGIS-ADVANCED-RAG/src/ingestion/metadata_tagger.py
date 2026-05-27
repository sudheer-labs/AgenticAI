from langchain_openai import ChatOpenAI
import re
import json
import os


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)


def extract_llm_metadata(chunk_text: str):

    prompt = f"""You're a Corprate policy document parser.
    Given a chunk of text from a corporate policy document, 
    your task is to extract the key information in a structured json format only and null if not available. 
    The output should include the following sections:
    
    1. document_id: Policy code or unique identifier for the document visible for example: LND-POL-7010-V3 else None.
    2. policy_category: One of the items [Travel, HR, IT, Security, Legal, Compliance, Finance, Other].
    3. policy_owner": Department Name if mentioned else None.
    4. effective_date: date in YYYY-MM-DD format if mentioned else None.

    TEXT: '''{chunk_text}'''
     
    """
    response = llm.invoke(prompt).content.strip()
    #clean the response to ensure it's valid JSON
    response = re.sub(r'```json|```', '', response).strip()
    try:
    # attempt something risky
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
    # if it fails, do this instead
        return {
        "document_id": None,
        "policy_category": "Other",
        "policy_owner": None,
        "effective_date": None
    }


def metadata_tagging(chunks: str):

    # Call the LLM with first chunk to extract metadata and then add the metadata to all chunks of the same document
    llm_metadata = extract_llm_metadata(chunks[0]["content"])
    for chunk in chunks:
        chunk["metadata"] = {**chunk["metadata"], **llm_metadata}
    return chunks