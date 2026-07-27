import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "regression"


def _fit_model(df):
    """Fit a random-intercept linear mixed model and return its results."""
    model = smf.mixedlm(
        "shortfall ~ n_rooms + age",
        data=df,
        groups=df["local_authority_code"],
    )
    return model.fit()


def _save_model_summary(model, outpath):
    with open(outpath, "w", encoding="utf-8") as file:
        file.write(model.summary().as_text())


def _random_effects(results):
    re_df = pd.DataFrame(results.random_effects).T
    re_df.columns = [
        "Intercept",
        *[
            f"Slope_{index}"
            for index in range(len(re_df.columns) - 1)
        ],
    ]
    re_df["group"] = re_df.index
    stderr = np.sqrt(results.cov_re.iloc[0, 0])
    re_df["lower"] = re_df["Intercept"] - 1.96 * stderr
    re_df["upper"] = re_df["Intercept"] + 1.96 * stderr
    re_df = re_df.sort_values("Intercept")
    return re_df


def fit_model():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / "data_cache" / "la_energy.csv")
    results = _fit_model(df)

    output_dir = base_dir / VIGNETTE_DIR
    model_dir = base_dir / "data_cache" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    _random_effects(results).to_csv(
        model_dir / "reffs.csv",
        index=False,
    )
    _save_model_summary(
        results,
        output_dir / "model_fit.txt",
    )
