import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_vectorstore(index_path: str = "faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(query: str, k: int = 3, vectorstore=None) -> list[str]:
    if vectorstore is None:
        return []
    results = vectorstore.similarity_search(query, k=k)
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]


def generate(
    query: str,
    retrieved_chunks: list[str],
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    prompt = f"""
You are an AI assistant answering questions about AuroraStay Hotels using
only the information provided in the retrieved hospitality documentation.

User Query:
{query}

Retrieved AuroraStay Documentation:
{retrieved_chunks}

Rules:
1. Answer the question using only information explicitly supported by the retrieved documentation.
2. Do not use outside knowledge or assumptions.
3. Answer directly, accurately, and completely, preserving rate plans, deadlines, deposits, check-in/out times, property names, room types, fees, accessibility rules, and escalation steps.
4. If the retrieved documentation does not contain enough information, respond exactly with:
   "The available documents do not contain enough information to answer this question."
5. If retrieved documentation contains conflicting information, identify the conflict clearly.

Answer:
"""
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def rag(
    query: str,
    k: int = 3,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None,
):
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return answer, retrieved_chunks
