import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from course.utils import find_project_root
from course.unsupervised_classification.tree import (
    _pca,
    _scatter_clusters,
)

VIGNETTE_DIR = (
    Path("data_cache")
    / "vignettes"
    / "unsupervised_classification"
)


def _kmeans(df, k):
    """Fit and return a KMeans model with k clusters."""
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    return model.fit(df)


def kmeans(k):
    base_dir = find_project_root()
    df = pd.read_csv(
        base_dir / "data_cache" / "la_collision.csv"
    )

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    model = _kmeans(df_scaled, k)
    clusters = model.labels_
    scaled_centers = model.cluster_centers_

    fig1, fig2 = _plot_centroids(
        scaled_centers,
        scaler,
        df.columns,
        k,
    )

    output_dir = base_dir / VIGNETTE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1.write_html(output_dir / "kcentroids1.html")
    fig2.write_html(output_dir / "kcentroids2.html")

    df_plot = _pca(df_scaled)
    df_plot["cluster"] = clusters.astype(str)

    fig = _scatter_clusters(df_plot)
    fig.write_html(output_dir / "kscatter.html")


def _plot_centroids(
    scaled_centers,
    scaler,
    colnames,
    k,
):
    original_centers = scaler.inverse_transform(
        scaled_centers
    )

    centers_df = pd.DataFrame(
        original_centers,
        columns=colnames,
    ).iloc[:, [0]]
    centers_df["cluster"] = [
        f"Cluster {index}"
        for index in range(k)
    ]

    centers_melted = centers_df.melt(
        id_vars="cluster",
        var_name="Feature",
        value_name="Value",
    )

    fig1 = px.bar(
        centers_melted,
        x="Feature",
        y="Value",
        color="cluster",
        barmode="group",
        title=(
            "Cluster Centers by Feature "
            "(Original Scale)"
        ),
    )

    centers_df = pd.DataFrame(
        original_centers,
        columns=colnames,
    ).iloc[:, 1:]
    centers_df["cluster"] = [
        f"Cluster {index}"
        for index in range(k)
    ]

    centers_melted = centers_df.melt(
        id_vars="cluster",
        var_name="Feature",
        value_name="Value",
    )

    fig2 = px.bar(
        centers_melted,
        x="Feature",
        y="Value",
        color="cluster",
        barmode="group",
        title=(
            "Cluster Centers by Feature "
            "(Original Scale)"
        ),
    )

    return fig1, fig2
