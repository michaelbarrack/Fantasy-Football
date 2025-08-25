# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# —— User parameters ——
FILE_PATH = "FF_FPROS_STATS.xlsx"
SHEET_NAME = "Stats"
POSITIONS = ["QB", "WR", "RB", "TE"]

st.set_page_config(layout="wide")
st.title("Fantasy Football PROJ_PPG Clustering (Top-N Only)")

@st.cache_data
def load_data():
    return pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

df_full = load_data()

for pos in POSITIONS:
    st.markdown(f"## {pos}")
    col_table, col_plot, col_settings = st.columns([2, 3, 1])

    # — SETTINGS (right) —
    with col_settings:
        df_pos_all = df_full[df_full["POS"] == pos]
        max_players = len(df_pos_all)

        top_n = st.number_input(
            f"Top _N_ for {pos}",
            min_value=1,
            max_value=max_players,
            value=min(10, max_players),
            key=f"{pos}_top_n"
        )
        n_clusters = st.number_input(
            f"Clusters _K_ for {pos}",
            min_value=1,
            max_value=top_n,
            value=min(5, top_n),
            key=f"{pos}_clusters"
        )

        # — NEW: which players to "remove" (mark with X)
        df_top_all = df_pos_all.nlargest(top_n, "PROJ_PPG")
        excluded = st.multiselect(
            f"Exclude {pos}s",
            options=df_top_all["Player"].tolist(),
            default=[],
            key=f"{pos}_exclude"
        )
        
        # Toggle to include/exclude excluded players from graphs
        include_excluded = st.toggle(
            f"Include excluded {pos}s in graph",
            value=True,
            key=f"{pos}_include_excluded"
        )

    # — 1) PICK TOP N —
    df_top = (
        df_pos_all
        .nlargest(top_n, "PROJ_PPG")
        .reset_index(drop=True)
        .copy()
    )
    df_top["rank"] = np.arange(1, len(df_top) + 1)
    df_top["excluded"] = df_top["Player"].isin(excluded)

    # — 2) CLUSTER —
    # Filter data for clustering based on toggle
    if include_excluded:
        df_for_clustering = df_top.copy()
    else:
        df_for_clustering = df_top[~df_top["excluded"]].copy()
    
    # Only proceed with clustering if we have enough data points
    if len(df_for_clustering) >= n_clusters:
        X = df_for_clustering[["PROJ_PPG"]].values
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
        df_for_clustering["cluster"] = kmeans.labels_

        # Compute each cluster's mean and order them
        cluster_means = df_for_clustering.groupby("cluster")["PROJ_PPG"].mean().to_dict()
        ordered = sorted(cluster_means, key=lambda c: cluster_means[c], reverse=True)
        label_map = {old: new for new, old in enumerate(ordered, start=1)}

        df_for_clustering["cluster_ordered"] = df_for_clustering["cluster"].map(label_map)
        df_for_clustering["cluster_mean"] = df_for_clustering["cluster"].map(cluster_means)
        
        # Merge clustering results back to the full dataframe
        df_top = df_top.merge(
            df_for_clustering[["Player", "cluster", "cluster_ordered", "cluster_mean"]], 
            on="Player", 
            how="left"
        )
    else:
        # Not enough data points for clustering, assign all to single cluster
        df_top["cluster"] = 0
        df_top["cluster_ordered"] = 1
        df_top["cluster_mean"] = df_for_clustering["PROJ_PPG"].mean() if len(df_for_clustering) > 0 else 0

    # — 3) TABLE (left) —
    with col_table:
        st.markdown(f"**Top {top_n} {pos}s by PROJ_PPG (Clustered)**")
        # show exclusion status too
        st.dataframe(
            df_top[["rank", "Player", "PROJ_PPG", "cluster_ordered", "excluded"]],
            use_container_width=True
        )

    # — 4) PLOT (middle) —
    with col_plot:
        fig, ax = plt.subplots()
        
        # Filter data for plotting based on toggle
        if include_excluded:
            df_for_plot = df_top.copy()
        else:
            df_for_plot = df_top[~df_top["excluded"]].copy()
        
        # Only plot if we have data
        if len(df_for_plot) > 0:
            # draw each cluster's mean line first
            for cl in sorted(df_for_plot["cluster_ordered"].dropna().unique()):
                dfc = df_for_plot[df_for_plot["cluster_ordered"] == cl]
                if len(dfc) > 0:
                    mean_ppg = dfc["cluster_mean"].iloc[0]
                    ax.hlines(
                        y=mean_ppg,
                        xmin=dfc["rank"].min(),
                        xmax=dfc["rank"].max(),
                        linestyles="--",
                        # get the color of a dummy plot so line matches markers below
                        colors=ax.plot([], [], label=f"Cluster {cl}: mean {mean_ppg:.2f}")[0].get_color()
                    )
            
            # now plot each point, X if excluded (when included in plot)
            for _, row in df_for_plot.iterrows():
                if pd.notna(row["cluster_ordered"]):  # Only plot if clustering was successful
                    marker = "X" if row["excluded"] else "o"
                    # find the same color used for its cluster line/label
                    # we can grab it back from the existing lines by matching label
                    label = f"Cluster {row['cluster_ordered']}: mean {row['cluster_mean']:.2f}"
                    line = next((l for l in ax.get_lines() if l.get_label() == label), None)
                    color = line.get_color() if line is not None else "black"
                    ax.scatter(
                        row["rank"],
                        row["PROJ_PPG"],
                        marker=marker,
                        s=100,
                        color=color
                    )

        ax.set_xlabel("Rank (1 = highest PROJ_PPG)")
        ax.set_ylabel("PROJ_PPG")
        ax.set_title(f"{pos}: PROJ_PPG vs. Rank")
        ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)

st.markdown("---")
st.caption("Clustering is now applied only to the Top-N players per position.  \nUse the toggle to include/exclude excluded players from graphs dynamically. When excluded players are included in the graph, they appear as X's.")
