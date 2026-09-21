import anthropic
from sentence_transformers import SentenceTransformer
import numpy as np

client = anthropic.Anthropic(api_key = "XXXXX")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Vert.x is a toolkit for building reactive applications on the JVM.",
    "GKE (Google Kubernetes Engine) is Google's managed Kubernetes service.",
    "GraphQL allows clients to request exactly the data they need from an API.",
]
doc_embeddings = embedder.encode(documents)

def retrieve(query, k=2):
    query_emb = embedder.encode([query])[0]
    scores = np.dot(doc_embeddings, query_emb) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    top_k_idx = np.argsort(scores)[-k:][::-1]
    return [documents[i] for i in top_k_idx]

def rag_agent(query):
    context = retrieve(query)
    prompt = f"""Use the following context to answer the question. If the context doesn't contain the answer, say so.

    Context:
    {chr(10).join(context)}
    
    Question: {query}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

print(rag_agent(input("What is the question you would like to ask?")))
