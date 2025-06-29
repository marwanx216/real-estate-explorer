import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def optimal_kmeans(df: pd.DataFrame, n_min=2, n_max=10):
    """
    Automatically determine the optimal number of clusters using silhouette score.
    """
    silhouettes = []
    models = []

    for k in range(n_min, n_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(df)
        score = silhouette_score(df, labels)
        silhouettes.append(score)
        models.append(model)

    best_index = int(np.argmax(silhouettes))
    best_model = models[best_index]
    best_k = best_model.n_clusters
    best_score = silhouettes[best_index]

    return best_model, best_k, best_score


def run_clustering(df: pd.DataFrame, nlp_cols: list, k_min: int = 2, k_max: int = 5):
    """
    Run clustering on the given DataFrame using NLP columns.
    Returns the updated DataFrame with 'Cluster', the model, and cluster stats.
    """
    if not nlp_cols or not all(col in df.columns for col in nlp_cols):
        raise ValueError("⚠️ NLP features not found. Please run the modeling step first.")

    features = df[nlp_cols]

    if features.empty:
        raise ValueError("❌ Provided NLP features are empty or invalid.")

    model, best_k, best_score = optimal_kmeans(features, n_min=k_min, n_max=k_max)

    df = df.copy()
    df["Cluster"] = model.predict(features)
    return df, model, best_k, best_score


def plot_clusters(df: pd.DataFrame, nlp_cols: list):
    """
    Create a 2D PCA visualization of the clusters.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df[nlp_cols])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=df["Cluster"], cmap="tab10", alpha=0.7)
    ax.set_title("Cluster Visualization (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    return fig
