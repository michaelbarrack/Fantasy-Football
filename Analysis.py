from PlayerData import *
import numpy


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = get_player_ppg()

players_ppg = query_player_ppg(df, position='WR', min_games=9)

X = players_ppg[["ppg"]].values          # feature matrix
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=0).fit(X_scaled)
players_ppg["cluster"] = kmeans.labels_


# ─── Order clusters by their overall mean PPG ─────────────────────
cluster_order = (players_ppg
                 .groupby("cluster")["ppg"]
                 .mean()
                 .sort_values(ascending=False)
                 .index)                       # e.g. Int64Index([2, 0, 1], dtype='int64')

# ─── Print each cluster’s roster, highest-PPG players first ───────
for cl in cluster_order:
    mean_ppg = players_ppg.loc[players_ppg.cluster == cl, "ppg"].mean()
    print(f"\n=== Cluster {cl}  (mean PPG: {mean_ppg:.2f}) ===")

    lines = (players_ppg            # build the table for this cluster …
           .loc[players_ppg.cluster == cl, ["player_name", "ppg"]]
           .sort_values("ppg", ascending=False)
           .to_string(index=False, header=["Player", "PPG"])
           .splitlines())

    print("\n".join("    " + line for line in lines))
