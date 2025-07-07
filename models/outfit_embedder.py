import os
import numpy as np
import pandas as pd
import faiss

def recommend_outfits(gender, style, top_k=9):
    """Recommends outfits based on gender and style by FAISS similarity search."""
    metadata = pd.read_csv("data/metadata.csv")
    metadata = metadata[(metadata["gender"] == gender) & (metadata["style"] == style)]

    if metadata.empty:
        return []

    # Load embeddings
    embeddings = np.load("data/embeddings.npy").astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    # Dummy query vector (this should ideally be selfie embedding later)
    query_vector = np.random.rand(embeddings.shape[1]).astype("float32")
    _, indices = faiss_index.search(query_vector.reshape(1, -1), top_k)

    recommended_paths = []
    for idx in indices[0]:
        if idx < len(metadata):
            recommended_paths.append(metadata.iloc[idx]["filename"])

    return recommended_paths
