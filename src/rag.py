class RAGPipeline:
    def __init__(self, vectordb, embedder):
        self.vectordb = vectordb
        self.embedder = embedder

    def retrieve(self, query: str, top_k=5):
        q_vec = self.embedder.embed([query])[0]
        results = self.vectordb.search(q_vec, top_k=top_k)

        matches = results.get("matches", [])
        contexts = []
        for m in matches:
            metadata = m.get("metadata", {})
            contexts.append({
                "text": metadata.get("text", ""),
                "source": metadata.get("source", "unknown"),
                "score": m.get("score", 0.0)
            })
        return contexts

    def generate_answer(self, query: str, contexts):
        if not contexts:
            return "I don't know based on the given documents."

        answer = "Answer (from retrieved context):\n\n"
        answer += "Here are the most relevant points found in your documents:\n\n"

        for i, c in enumerate(contexts[:3], 1):
            answer += f"{i}. {c['text']}\n\n"

        answer += "\n📌 Note: This answer is generated without an external LLM (Groq), using retrieved context only."
        return answer
