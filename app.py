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

        # — NEW: which players to “remove” (mark with X)
        df_top_all = df_pos_all.nlargest(top_n, "PROJ_PPG")
        excluded = st.multiselect(
            f"Exclude {pos}s",
            options=df_top_all["Player"].tolist(),
            default=[],
            key=f"{pos}_exclude"
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

    # — 2) CLUSTER ONLY ON TOP N —
    X = df_top[["PROJ_PPG"]].values
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
    df_top["cluster"] = kmeans.labels_

    # Compute each cluster's mean and order them
    cluster_means = df_top.groupby("cluster")["PROJ_PPG"].mean().to_dict()
    ordered = sorted(cluster_means, key=lambda c: cluster_means[c], reverse=True)
    label_map = {old: new for new, old in enumerate(ordered, start=1)}

    df_top["cluster_ordered"] = df_top["cluster"].map(label_map)
    df_top["cluster_mean"]    = df_top["cluster"].map(cluster_means)

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
        # draw each cluster’s mean line first
        for cl in sorted(df_top["cluster_ordered"].unique()):
            dfc = df_top[df_top["cluster_ordered"] == cl]
            mean_ppg = dfc["cluster_mean"].iloc[0]
            ax.hlines(
                y=mean_ppg,
                xmin=dfc["rank"].min(),
                xmax=dfc["rank"].max(),
                linestyles="--",
                # get the color of a dummy plot so line matches markers below
                colors=ax.plot([], [], label=f"Cluster {cl}: mean {mean_ppg:.2f}")[0].get_color()
            )
        # now plot each point, X if excluded
        for _, row in df_top.iterrows():
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
st.caption("Clustering is now applied only to the Top-N players per position.  \nExcluded players appear as X’s on the plot but remain in the table and cluster calculations.")
