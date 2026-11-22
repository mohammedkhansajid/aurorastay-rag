from flask import Flask, request, jsonify
from rag_utils import get_vectorstore, retrieve, rag, DEFAULT_MODEL

app = Flask(__name__)
vectorstore = get_vectorstore("faiss_index")


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/v1/relevant_chunks")
def get_relevant_chunks():
    data = request.get_json(force=True)
    query = data.get("query", "")
    k = int(data.get("k", 3))
    chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    return jsonify({"query": query, "k": k, "relevant_chunks": chunks})


@app.post("/v1/answer_with_relevant_chunks")
def get_answer_with_relevant_chunks():
    data = request.get_json(force=True)
    query = data.get("query", "")
    k = int(data.get("k", 3))
    model_name = data.get("model_name", DEFAULT_MODEL)
    temperature = float(data.get("temperature", 0.0))
    top_p = float(data.get("top_p", 1.0))
    max_tokens = int(data.get("max_tokens", 512))

    answer, chunks = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore,
    )
    return jsonify({
        "query": query,
        "answer": answer,
        "relevant_chunks": chunks,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
