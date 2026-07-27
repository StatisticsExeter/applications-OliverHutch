import joblib
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from course.utils import find_project_root


def fit_classifier(
    X_train_path,
    y_train_path,
    model_path,
    classifier,
):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)["built_age"]

    classifier.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_path)


def fit_lda():
    base_dir = find_project_root()
    X_train_path = (
        base_dir / "data_cache" / "energy_X_train.csv"
    )
    y_train_path = (
        base_dir / "data_cache" / "energy_y_train.csv"
    )
    model_path = (
        base_dir
        / "data_cache"
        / "models"
        / "lda_model.joblib"
    )

    classifier = LinearDiscriminantAnalysis()
    fit_classifier(
        X_train_path,
        y_train_path,
        model_path,
        classifier,
    )


def fit_qda():
    base_dir = find_project_root()
    X_train_path = (
        base_dir / "data_cache" / "energy_X_train.csv"
    )
    y_train_path = (
        base_dir / "data_cache" / "energy_y_train.csv"
    )
    model_path = (
        base_dir
        / "data_cache"
        / "models"
        / "qda_model.joblib"
    )

    classifier = QuadraticDiscriminantAnalysis()
    fit_classifier(
        X_train_path,
        y_train_path,
        model_path,
        classifier,
    )
