# pipeline/optimization.py

import optuna
import numpy as np
import logging

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from pipeline.modeling import detect_problem_type, get_pipelines

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ESPACIOS DE BÚSQUEDA OPTIMIZADOS
# ─────────────────────────────────────────────

def get_search_space(trial, model_name, problem_type):

    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        }

    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", -1, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 0.1, 20.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    if model_name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }

    if model_name == "MLP":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [(64,), (128,), (128, 64)]
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        }

    if model_name == "DecisionTree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }

    return {}


# ─────────────────────────────────────────────
# FUNCIÓN OBJETIVO OPTIMIZADA
# ─────────────────────────────────────────────

def objective(trial, X, y, model_name):

    problem_type = detect_problem_type(y)

    # fast_mode=True → pipelines más ligeros para Optuna
    pipelines = get_pipelines(
        n_features=X.shape[1],
        problem_type=problem_type,
        fast_mode=True
    )

    pipe = pipelines[model_name]

    params = get_search_space(trial, model_name, problem_type)
    params = {f"model__{k}": v for k, v in params.items()}

    pipe.set_params(**params)

    # CV reducido para velocidad
    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = "f1_weighted"
    else:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scoring = "r2"

    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return float(np.mean(scores))


# ─────────────────────────────────────────────
# OPTIMIZACIÓN EFICIENTE TOP-K
# ─────────────────────────────────────────────

def optimize_top_models(
    X,
    y,
    top_models,
    n_trials=25,
    timeout=600
):
    """
    Optimiza SOLO los modelos TOP seleccionados.

    Estrategia:
    - Reduce costo computacional
    - Usa pruning agresivo
    - Usa pipelines rápidos
    """

    best_results = []

    for model_name in top_models:

        logger.info(f"Optimizando {model_name} con Optuna...")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5
            )
        )

        study.optimize(
            lambda trial: objective(trial, X, y, model_name),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1  # importante para evitar sobrecarga
        )

        best_results.append({
            "model": model_name,
            "best_score": float(study.best_value),
            "best_params": study.best_params
        })

        logger.info(
            f"{model_name} → best_score={study.best_value:.4f}"
        )

    # ordenar resultados
    best_results = sorted(
        best_results,
        key=lambda x: x["best_score"],
        reverse=True
    )

    return best_results