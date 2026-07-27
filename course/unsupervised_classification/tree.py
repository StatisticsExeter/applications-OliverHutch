import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from course.utils import find_project_root

VIGNETTE_DIR = (
    Path("data_cache")
    / "vignettes"
    / "unsupervised_classification"
)


def hcluster_analysis():
    base_dir = find_project_root()
    df = pd.read_csv(
        base_dir / "data_cache" / "la_collision.csv"
    )

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    output_dir = base_dir / VIGNETTE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = _plot_dendrogram(df_scaled)
    fig.write_html(output_dir / "dendrogram.html")


def hierarchical_groups(height):
    base_dir = find_project_root()
    df = pd.read_csv(
        base_dir / "data_cache" / "la_collision.csv"
    )

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    linked = _fit_dendrogram(df_scaled)
    clusters = _cutree(linked, height)

    df_plot = _pca(df_scaled)
    df_plot["cluster"] = (
        clusters["cluster"].astype(str).to_numpy()
    )

    output_dir = base_dir / VIGNETTE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = _scatter_clusters(df_plot)
    fig.write_html(output_dir / "hscatter.html")


def _fit_dendrogram(df):
    """Return a Ward hierarchical clustering solution."""
    return linkage(df, method="ward")


def _plot_dendrogram(df):
    """Return a Plotly dendrogram for the supplied data."""
    fig = ff.create_dendrogram(
        df,
        linkagefun=lambda values: linkage(
            values,
            method="ward",
        ),
    )
    fig.update_layout(
        title="Interactive Hierarchical Clustering Dendrogram"
    )
    return fig


def _cutree(tree, height):
    """Cut a hierarchy at height and return memberships."""
    clusters = fcluster(
        tree,
        t=height,
        criterion="distance",
    )
    return pd.DataFrame({"cluster": clusters})


def _pca(df):
    """Return the first two principal-component scores."""
    values = PCA(n_components=2).fit_transform(df)

    return pd.DataFrame(
        values,
        columns=["PC1", "PC2"],
    )


def _scatter_clusters(df):
    """Return a PC1-versus-PC2 cluster scatterplot."""
    return px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="PCA Scatter Plot Colored by Cluster Labels",
    )
