import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_faiss(chunks, embedder):
    embeddings = embedder.encode([chunk['content'] for chunk in chunks], show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Step 1: Load gdpr_articles.json
    articles_path = "data/gdpr_articles.json"
    with open(articles_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    print(f"✅ Loaded {len(all_chunks)} chunks from gdpr_articles.json")

    # Step 2: Embed and build FAISS
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = build_faiss(all_chunks, embedder)

    # Step 3: Save FAISS index
    faiss.write_index(index, "data/faiss.index")
    print("✅ FAISS index saved under /data/faiss.index")
