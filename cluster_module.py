# cluster_module.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_position_clusters_from_excel(
    file_path: str,
    sheet_name: str = "Stats",
    position: str = None,
    n_clusters: int = 5
) -> pd.DataFrame:
    """
    Reads the Excel file, filters by position (if provided),
    clusters on PROJ_PPG into n_clusters groups, then
    orders clusters by descending mean PROJ_PPG.

    Returns a DataFrame with columns:
      - all original columns
      - cluster            (0..n_clusters-1, labels from KMeans)
      - cluster_ordered    (1 = highest-mean cluster, …)
      - cluster_mean       (mean PROJ_PPG for that cluster)
    """
    # 1) Load and filter
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if position is not None:
        df = df[df["POS"] == position].copy()
    else:
        df = df.copy()

    # 2) Extract values and scale
    X = df[["PROJ_PPG"]].values
    X_scaled = StandardScaler().fit_transform(X)

    # 3) K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # 4) Compute each cluster’s mean (on original scale)
    cluster_means = df.groupby("cluster")["PROJ_PPG"].mean().to_dict()

    # 5) Order clusters by descending mean
    ordered = sorted(cluster_means, key=lambda c: cluster_means[c], reverse=True)
    # map old label → new “ranked” label (1 = highest-mean)
    cluster_map = {old: new for new, old in enumerate(ordered, start=1)}

    df["cluster_ordered"] = df["cluster"].map(cluster_map)
    df["cluster_mean"] = df["cluster"].map(cluster_means)

    return df