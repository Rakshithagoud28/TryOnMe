import pandas as pd

def recommend_outfits(gender, style, top_k=9):
    """Recommends outfits based on gender and style by metadata filtering."""
    # Map lowercase style (from UI) to capitalized style (in CSV)
    style_map = {
        "casual": "Casual",
        "formal": "Formal",
        "party": "Party"
    }

    mapped_style = style_map.get(style.lower(), style)

    metadata = pd.read_csv("data/metadata.csv")
    filtered = metadata[(metadata["gender"] == gender) & (metadata["style"] == mapped_style)]

    print("GENDER:", gender)
    print("UI STYLE:", style)
    print("MAPPED STYLE:", mapped_style)
    print("Found outfits:", filtered.shape[0])

    if filtered.empty:
        return []

    # Just return filenames for now (skip FAISS until verified)
    return filtered["filename"].tolist()
