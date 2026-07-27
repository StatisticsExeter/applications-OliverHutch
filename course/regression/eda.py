import pandas as pd
import plotly.express as px
from pathlib import Path

from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "regression"


def _boxplot(df, x_var, y_var, title):
    """Return a Plotly box plot for the requested columns."""
    return px.box(
        df,
        x=x_var,
        y=y_var,
        title=title,
    )


def boxplot_age():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / "data_cache" / "la_energy.csv")
    fig = _boxplot(
        df,
        "age",
        "shortfall",
        "Shortfall by Age Category",
    )
    fig.write_html(base_dir / VIGNETTE_DIR / "boxplot_age.html")


def boxplot_rooms():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / "data_cache" / "la_energy.csv")
    fig = _boxplot(
        df,
        "n_rooms",
        "shortfall",
        "Shortfall by Number of rooms",
    )
    fig.write_html(base_dir / VIGNETTE_DIR / "boxplot_rooms.html")
