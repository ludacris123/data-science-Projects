"""Segment retail customers with K-Means and profile each cluster."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
FEATURES = ["annual_income", "spending_score", "purchase_frequency", "average_order_value"]


def make_customers(rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    centers = np.array([
        [32000, 28, 4, 35],
        [62000, 62, 12, 85],
        [105000, 82, 22, 165],
        [88000, 35, 7, 110],
    ])
    spread = np.array([9000, 11, 2.5, 18])
    groups = rng.choice(len(centers), rows, p=[0.30, 0.35, 0.20, 0.15])
    values = centers[groups] + rng.normal(size=(rows, 4)) * spread
    values[:, 0] = values[:, 0].clip(15000, 180000)
    values[:, 1] = values[:, 1].clip(1, 100)
    values[:, 2] = values[:, 2].clip(1, 40)
    values[:, 3] = values[:, 3].clip(10, 300)
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, "customer_id", [f"C{i:05d}" for i in range(1, rows + 1)])
    return frame.round(2)


def persona(row: pd.Series) -> str:
    if row["spending_score"] >= 65 and row["annual_income"] >= 75000:
        return "Premium Loyalists"
    if row["purchase_frequency"] >= 10:
        return "Frequent Shoppers"
    if row["annual_income"] >= 75000 and row["spending_score"] < 50:
        return "High-Potential Customers"
    return "Value Seekers"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    customers = make_customers()
    scaled = StandardScaler().fit_transform(customers[FEATURES])

    scores = {}
    models = {}
    for clusters in range(2, 7):
        model = KMeans(n_clusters=clusters, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        scores[clusters] = silhouette_score(scaled, labels)
        models[clusters] = model

    best_k = max(scores, key=scores.get)
    labels = models[best_k].labels_
    customers["cluster"] = labels

    summary = customers.groupby("cluster")[FEATURES].mean().round(2)
    summary["customers"] = customers.groupby("cluster").size()
    summary["persona"] = summary.apply(persona, axis=1)
    persona_map = summary["persona"].to_dict()
    customers["persona"] = customers["cluster"].map(persona_map)

    customers.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "segment_profiles.csv")
    (OUTPUT_DIR / "model_selection.json").write_text(json.dumps({
        "selected_clusters": best_k,
        "silhouette_scores": {str(k): round(v, 4) for k, v in scores.items()},
    }, indent=2))

    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="viridis", alpha=0.65)
    plt.title(f"Customer Segments (K={best_k})")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segments_pca.png", dpi=160)
    plt.close()
    print(summary.to_string())


if __name__ == "__main__":
    main()
