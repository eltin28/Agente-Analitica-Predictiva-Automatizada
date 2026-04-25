# pipeline/modeling.py

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

# Modelos
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ─────────────────────────────────────────────
# FACTORY (OCP)
# ─────────────────────────────────────────────

def get_model_instance(model_name: str, problem_type: str):
    """
    Devuelve una instancia NUEVA del modelo.
    Nunca reutilizar instancias (evita efectos colaterales).

    LightGBM usa verbose=-1 para suprimir logs internos y
    feature_name='auto' para no depender de nombres de columnas
    (evita el warning "X does not have valid feature names").
    MLP usa max_iter=1000 para garantizar convergencia.
    """

    if problem_type == "classification":
        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "MLP": MLPClassifier(
                        max_iter=1000,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=42
                    ),
        }

        if HAS_LGBM:
            models["LightGBM"] = LGBMClassifier(
                random_state=42,
                verbose=-1,          # silencia todos los logs de LightGBM
            )

    else:
        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "SVM": SVR(),
            "KNN": KNeighborsRegressor(),
            "MLP": MLPRegressor(
                        max_iter=1000,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=42
                    ),
        }

        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(
                random_state=42,
                verbose=-1,
            )

    if model_name not in models:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return models[model_name]


def get_models(problem_type: str):
    """
    Retorna lista de nombres de modelos disponibles.
    """
    base = ["RandomForest", "DecisionTree", "SVM", "KNN", "MLP"]
    if HAS_LGBM:
        base.append("LightGBM")
    return base


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def build_pipeline(model, preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_single_model(X, y, model_name, problem_type, preprocessor, params=None):
    model = get_model_instance(model_name, problem_type)

    if params:
        model.set_params(**params)

    pipeline = build_pipeline(model, preprocessor)
    pipeline.fit(X, y)

    return pipeline


def train_models(X, y, model_names, problem_type, preprocessor):
    trained = {}

    for name in model_names:
        pipeline = train_single_model(
            X, y, name, problem_type, preprocessor
        )
        trained[name] = pipeline

    return trained


def train_model_with_params(X, y, model_name, params, problem_type, preprocessor):
    pipeline = train_single_model(
        X, y, model_name, problem_type, preprocessor, params
    )
    return {model_name: pipeline}


# ─────────────────────────────────────────────
# EVALUACIÓN (CV)
# ─────────────────────────────────────────────

def _get_cv(problem_type):
    if problem_type == "classification":
        return StratifiedKFold(5, shuffle=True, random_state=42), "f1_weighted"
    else:
        return KFold(5, shuffle=True, random_state=42), "r2"


def evaluate_models(X, y, problem_type, preprocessor):
    results = []

    cv, scoring = _get_cv(problem_type)

    for model_name in get_models(problem_type):
        model = get_model_instance(model_name, problem_type)

        pipeline = build_pipeline(model, clone(preprocessor))

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1  # importante para multiprocessing
        )

        results.append({
            "model": model_name,
            "mean_score": scores.mean(),
            "std_score": scores.std()
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="mean_score", ascending=False)

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# EVALUACIÓN ROBUSTA (penaliza varianza)
# ─────────────────────────────────────────────

def evaluate_models_robust(X, y, model_names, problem_type, preprocessor):
    results = []

    cv, scoring = _get_cv(problem_type)

    for model_name in model_names:
        model = get_model_instance(model_name, problem_type)
        pipeline = build_pipeline(model, clone(preprocessor))

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )

        mean = scores.mean()
        std = scores.std()

        # Penalización simple
        robust_score = mean - std

        results.append({
            "model": model_name,
            "mean_score": mean,
            "std_score": std,
            "robust_score": robust_score
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="robust_score", ascending=False)

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# EVALUACIÓN FINAL (TEST)
# ─────────────────────────────────────────────

def evaluate_trained_models(models_dict, X_test, y_test, problem_type):
    from sklearn.metrics import (
        accuracy_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score
    )

    results = []

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            results.append({
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
            })
        else:
            results.append({
                "model": name,
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            })

    return pd.DataFrame(results)