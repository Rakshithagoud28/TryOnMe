import numpy as np
import pandas as pd

# Load metadata
metadata = pd.read_csv("data/metadata.csv")

# Dummy embedding generation (simulate image feature vectors)
num_images = len(metadata)
embedding_dim = 512  # Example embedding dimension (CLIP/ResNet-like)

# Random dummy vectors for testing
embeddings = np.random.rand(num_images, embedding_dim).astype("float32")

# Save embeddings
np.save("data/embeddings.npy", embeddings)

print(f"✅ Generated dummy embeddings for {num_images} images.")
