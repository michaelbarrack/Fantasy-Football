from PlayerData import *
import numpy


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_position_clusters(df : pd.DataFrame, position : str, min_games : int = 0, n_clusters : int = 5):

       players_ppg = query_player_ppg(df, position=position, min_games=min_games)

       X = players_ppg[["ppg"]].values
       X_scaled = StandardScaler().fit_transform(X)

       kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
       players_ppg["cluster"] = kmeans.labels_


       # ─── Order clusters by their overall mean PPG ─────────────────────
       cluster_order = (players_ppg
                     .groupby("cluster")["ppg"]
                     .mean()
                     .sort_values(ascending=False)
                     .index)
       
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


df = get_player_ppg()


get_position_clusters(df, position='RB', min_games=7, n_clusters=7)