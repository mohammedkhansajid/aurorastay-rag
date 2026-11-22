import os
import requests
import streamlit as st

st.set_page_config(page_title="AuroraStay Internal AI Assistant", layout="wide")
st.title("AuroraStay Hotels — Internal Policy AI Assistant")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:7860").rstrip("/")

with st.sidebar:
    st.header("Retrieval & Generation Settings")
    backend_url = st.text_input("Backend API URL", value=BACKEND_URL).rstrip("/")
    k = st.slider("Top-K Chunks (k)", min_value=1, max_value=8, value=3)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
    max_tokens = st.number_input("Max Tokens", min_value=128, max_value=2048, value=512, step=64)

query = st.text_area(
    "Enter guest policy or operational question:",
    placeholder="e.g., What is the cancellation policy for flexible rate reservations, and are there any deposit requirements?",
)

if st.button("Ask Assistant", type="primary") and query.strip():
    with st.spinner("Retrieving AuroraStay policies and generating response..."):
        try:
            res = requests.post(
                f"{backend_url}/v1/answer_with_relevant_chunks",
                json={
                    "query": query,
                    "k": k,
                    "model_name": "openai/gpt-oss-20b",
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": int(max_tokens),
                },
                timeout=90,
            )
            res.raise_for_status()
            data = res.json()
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            with st.expander("Supporting Policy Chunks"):
                for idx, chunk in enumerate(data.get("relevant_chunks", []), 1):
                    st.markdown(f"**Chunk {idx}**")
                    st.code(chunk)
        except Exception as exc:
            st.error(f"Error calling backend API: {exc}")
