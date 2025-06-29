# nlp_features.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Load transformer model once (fast and efficient)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text_column(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """
    Generate sentence embeddings for a given text column using SentenceTransformer.
    """
    print(f"[NLP] Embedding '{column}' with SBERT...")
    texts = df[column].fillna("").astype(str).tolist()
    embeddings = sbert_model.encode(texts, show_progress_bar=True)
    embed_df = pd.DataFrame(embeddings, columns=[f"{prefix}_emb_{i}" for i in range(embeddings.shape[1])])
    return embed_df


def tfidf_features(df: pd.DataFrame, column: str, prefix: str, max_features=50) -> pd.DataFrame:
    """
    Generate TF-IDF vectors for a given text column.
    """
    print(f"[NLP] Generating TF-IDF features for '{column}'...")
    texts = df[column].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(texts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{prefix}_tfidf_{w}" for w in tfidf.get_feature_names_out()])
    return tfidf_df


def enrich_with_nlp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: add both SBERT and TF-IDF for Title and Location columns.
    Returns a DataFrame with original + new features.
    """
    print("[NLP] Enriching DataFrame with NLP features...")

    # Embeddings
    location_embed = embed_text_column(df, "Location", "loc")
    title_embed = embed_text_column(df, "Title", "title")

    # TF-IDF (optional, but useful for trees)
    location_tfidf = tfidf_features(df, "Location", "loc", max_features=30)
    title_tfidf = tfidf_features(df, "Title", "title", max_features=30)

    # Combine all
    df = pd.concat([df.reset_index(drop=True),
                    location_embed, title_embed,
                    location_tfidf, title_tfidf], axis=1)

    print("[NLP] NLP enrichment complete.")
    return df
