import streamlit as st
import requests
import uuid

# Configuration - Change this to your live Render FastAPI URL after deploying it
BACKEND_URL = st.sidebar.text_input(
    "Backend API URL", 
    value="http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="AEGIS Policy Assistant", 
    page_icon="🛡️", 
    layout="wide"
)

# ─── ARCHITECTURE & TOPICS INFO SECTION ──────────────────────────────────
# This acts as an info dropdown right at the top of the interface
with st.expander("ℹ️ About AEGIS RAG Architecture & Pipeline Topics", expanded=False):
    st.markdown("""
    ### Advanced RAG Pipeline Architecture
    This chatbot utilizes a multi-stage **Retrieval-Augmented Generation (RAG)** pipeline optimized for production-grade document search and reasoning. Below are the core topics and techniques implemented in `chat.py`:
    
    *   **🎯 Intent & Category Detection:** Queries are pre-evaluated by an LLM to automatically isolate and pre-route requests according to structural metadata boundaries.
    *   **🔄 Multi-Query Expansion:** Generates multiple semantic variations of your query to prevent narrow, fragile search matches caused by phrasing choices.
    *   **🧠 Hypothetical Document Embeddings (HyDE):** An LLM drafts an artificial 'ideal answer' to your question. The system then embeds and searches against that draft to drastically match real document contexts better.
    *   **🛠️ Metadata Pre-Filtering:** Hard filters are executed inside the Weaviate vector database prior to vector search to prune irrelevant categorical segments instantly.
    *   **🔀 Reciprocal Rank Fusion (RRF):** Evaluates multi-query list configurations together, scoring and merging them using the $\\frac{1}{60 + \\text{rank}}$ reciprocal equation for robust retrieval.
    *   **⏳ Post-Retrieval Date Filtering:** Ensures temporal alignment by dropping documentation out of date boundaries after vector processing.
    *   **🚀 Cross-Encoder Reranking:** Leverages a deep, secondary transformer model to compute cross-attention text relevance scores between the question and top chunks, refining accuracy down to the top 5 results.
    *   **🪙 Dynamic Token Budgeting:** Manages context safely by calculating token sizes and truncating trailing documents to stay within a strict 3,000-token threshold.
    *   **💾 Stateful Conversational History:** Uses state stores to track linear human/assistant messages inside an active conversational buffer memory.
    """)

st.markdown("---")

st.title("🛡️ AEGIS Policy Assistant")
st.caption("AI-powered assistant utilizing advanced retrieval and reranking strategies to answer corporate policy queries.")

# Initialize unique session ID for the user's chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history for UI rendering
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for metadata and advanced pipeline observability
with st.sidebar:
    st.header("Pipeline Diagnostics")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Session ID:**")
    st.code(st.session_state.session_id, language="text")

# Render existing chat conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If there are sources or diagnostics saved in the state, render them
        if message["role"] == "assistant" and "diagnostics" in message:
            diag = message["diagnostics"]
            
            # Category display
            if diag.get("category"):
                st.caption(f"**Detected Category:** {diag['category']}")
                
            # Concepts used list
            if diag.get("concepts"):
                st.caption(f"**Pipeline Concepts Applied:** {', '.join(diag['concepts'])}")
            
            # Sources expander
            if diag.get("sources"):
                with st.expander("📚 View Document Sources"):
                    for idx, src in enumerate(diag["sources"]):
                        st.markdown(f"**Source {idx+1}: {src['document_id']}** (Relevance: {src['score']})")
                        if src.get("section"):
                            st.markdown(f"*Section:* {src['section']}")
                        st.text_area(f"Preview (Chunk {idx+1})", src["text_preview"], height=100, disabled=True)

            # Token info metric
            if diag.get("token_info"):
                ti = diag["token_info"]
                with st.expander("🪙 Token Budget Diagnostics"):
                    col1, col2 = st.columns(2)
                    col1.metric("Before Chunks", ti["chunks_before"])
                    col2.metric("After Budget Chunks", ti["chunks_after"])
                    st.json(ti)

# Handle user interaction
if user_query := st.chat_input("Ask a policy question..."):
    # Display user input
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Call FastAPI Backend
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Processing through AEGIS RAG pipeline..."):
            try:
                payload = {
                    "query": user_query,
                    "session_id": st.session_state.session_id
                }
                res = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=45)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    
                    # Output response
                    response_placeholder.markdown(answer)
                    
                    # Store information for diagnostics rendering
                    diagnostics = {
                        "category": data.get("category_detected"),
                        "concepts": data.get("concepts_used", []),
                        "sources": data.get("sources", []),
                        "token_info": data.get("token_info")
                    }
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "diagnostics": diagnostics
                    })
                    st.rerun()

                else:
                    response_placeholder.error(f"Backend Error ({res.status_code}): {res.text}")
            except requests.exceptions.RequestException as e:
                response_placeholder.error(f"Could not connect to FastAPI Backend: {e}")