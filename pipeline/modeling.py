# pipeline/modeling.py

import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from pipeline.config import (
    detect_execution_mode,
    MODEL_CONFIG,
    SMOTE_CONFIG,
    FEATURE_SELECTION_CONFIG
)

# Clasificación
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Regresión
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import lightgbm as lgb

from pipeline.preprocessing import CorrelationFilter
from pipeline.utils import detect_problem_type

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURACIONES RÁPIDAS vs FINALES
# ─────────────────────────────────────────────

def _get_model(model_name, problem_type, fast_mode=True):

    if problem_type == "classification":

        if model_name == "DecisionTree":
            return DecisionTreeClassifier(random_state=42)

        if model_name == "KNN":
            return KNeighborsClassifier(n_neighbors=5)

        if model_name == "SVM":
            return SVC(C=1.0, random_state=42)

        if model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=100 if fast_mode else 300,
                max_depth=10 if fast_mode else None,
                n_jobs=-1,
                random_state=42
            )

        if model_name == "LightGBM":
            return lgb.LGBMClassifier(
                n_estimators=100 if fast_mode else 300,
                learning_rate=0.1 if fast_mode else 0.05,
                random_state=42,
                verbose=-1
            )

        if model_name == "MLP":
            return MLPClassifier(
                max_iter=200 if fast_mode else 500,
                early_stopping=True,
                random_state=42
            )

    else:

        if model_name == "DecisionTree":
            return DecisionTreeRegressor(random_state=42)

        if model_name == "KNN":
            return KNeighborsRegressor(n_neighbors=5)

        if model_name == "SVM":
            return SVR(C=1.0)

        if model_name == "RandomForest":
            return RandomForestRegressor(
                n_estimators=100 if fast_mode else 300,
                max_depth=10 if fast_mode else None,
                n_jobs=-1,
                random_state=42
            )

        if model_name == "LightGBM":
            return lgb.LGBMRegressor(
                n_estimators=100 if fast_mode else 300,
                learning_rate=0.1 if fast_mode else 0.05,
                random_state=42,
                verbose=-1
            )

        if model_name == "MLP":
            return MLPRegressor(
                max_iter=200 if fast_mode else 500,
                early_stopping=True,
                random_state=42
            )


# ─────────────────────────────────────────────
# PIPELINES
# ─────────────────────────────────────────────

def get_pipelines(n_features, problem_type, y=None, fast_mode=True):

    # Validar y para evitar error en detect_execution_mode
    n_samples = len(y) if y is not None else 1000
    mode = detect_execution_mode(n_samples)
    config = MODEL_CONFIG.get(mode, MODEL_CONFIG.get(list(MODEL_CONFIG.keys())[0]))

    model_names = config["models"] if config else ["DecisionTree", "KNN", "RandomForest"]

    # Feature selection dinámico
    k = max(
        FEATURE_SELECTION_CONFIG["min_features"],
        int(n_features * FEATURE_SELECTION_CONFIG["ratio"])
    )
    k = min(k, n_features)

    pipelines = {}

    for name in model_names:

        steps = [("corr_filter", CorrelationFilter())]

        # ── CLASIFICACIÓN ───────────────────────────
        if problem_type == "classification":

            # SMOTE inteligente
            if (
                y is not None
                and SMOTE_CONFIG["enabled"]
                and (not fast_mode or SMOTE_CONFIG["apply_in_fast_phase"])
                and _needs_smote(y, SMOTE_CONFIG["imbalance_threshold"])
            ):
                steps.append(("smote", SMOTE(random_state=42)))

            # Feature selection selectiva
            if (
                FEATURE_SELECTION_CONFIG["enabled"]
                and name in FEATURE_SELECTION_CONFIG["apply_to_models"]
            ):
                steps.append(("select", SelectKBest(f_classif, k=k)))

        # ── REGRESIÓN ───────────────────────────────
        else:
            if (
                FEATURE_SELECTION_CONFIG["enabled"]
                and name in FEATURE_SELECTION_CONFIG["apply_to_models"]
            ):
                steps.append(("select", SelectKBest(f_regression, k=k)))

        steps.append(("model", _get_model(name, problem_type, fast_mode)))

        pipelines[name] = ImbPipeline(steps)

    return pipelines

def _needs_smote(y, threshold=0.2, min_samples=6):
    values, counts = np.unique(y, return_counts=True)
    ratios = counts / counts.sum()

    # Evitar errores internos de SMOTE (k_neighbors=5)
    if counts.min() < min_samples:
        return False

    return ratios.min() < threshold

# ─────────────────────────────────────────────
# EVALUACIÓN OPTIMIZADA
# ─────────────────────────────────────────────

def evaluate_models(X_train, y_train, problem_type):

    n_features = X_train.shape[1]

    mode = detect_execution_mode(len(y_train))
    config = MODEL_CONFIG[mode]

    # ── FASE 1: rápida ───────────────────────────
    pipelines_fast = get_pipelines(
        n_features, problem_type, y_train, fast_mode=True
    )

    cv_fast = (
        StratifiedKFold(config["cv_folds_fast"], shuffle=True, random_state=42)
        if problem_type == "classification"
        else KFold(config["cv_folds_fast"], shuffle=True, random_state=42)
    )

    scoring = (
        {"f1": "f1_weighted", "acc": "accuracy"}
        if problem_type == "classification"
        else {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
    )

    fast_results = []

    for name, pipe in pipelines_fast.items():
        try:
            scores = cross_validate(
                pipe, X_train, y_train, cv=cv_fast, scoring=scoring, n_jobs=1
            )

            metric = (
                scores["test_f1"].mean()
                if problem_type == "classification"
                else scores["test_r2"].mean()
            )

            fast_results.append((name, metric))

        except Exception as e:
            logger.warning(f"{name} falló en fast CV: {e}")

    fast_results.sort(key=lambda x: x[1], reverse=True)
    top_models = [m[0] for m in fast_results[:config["top_k_models"]]]

    # ── FASE 2: completa ─────────────────────────
    pipelines_full = get_pipelines(
        n_features, problem_type, y_train, fast_mode=False
    )

    cv_full = (
        StratifiedKFold(config["cv_folds_full"], shuffle=True, random_state=42)
        if problem_type == "classification"
        else KFold(config["cv_folds_full"], shuffle=True, random_state=42)
    )

    results = []

    for name in top_models:
        pipe = pipelines_full[name]

        try:
            scores = cross_validate(
                pipe, X_train, y_train, cv=cv_full, scoring=scoring, n_jobs=1
            )

            row = {"model": name}

            if problem_type == "classification":
                row["f1_mean"] = scores["test_f1"].mean()
                row["accuracy_mean"] = scores["test_acc"].mean()
            else:
                row["r2_mean"] = scores["test_r2"].mean()
                row["rmse_mean"] = -scores["test_rmse"].mean()

            results.append(row)

        except Exception as e:
            logger.warning(f"{name} falló en full CV: {e}")

    return pd.DataFrame(results).sort_values(
        by="f1_mean" if problem_type == "classification" else "r2_mean",
        ascending=False
    )


# ─────────────────────────────────────────────
# ENTRENAMIENTO FINAL
# ─────────────────────────────────────────────

def train_models(X_train, y_train, selected_models, problem_type):

    pipelines = get_pipelines(
        X_train.shape[1],
        problem_type,
        y_train,
        fast_mode=False
    )

    trained = {}

    for name in selected_models:
        try:
            pipe = pipelines[name]
            pipe.fit(X_train, y_train)
            trained[name] = pipe
        except Exception as e:
            logger.warning(f"{name} falló entrenamiento: {e}")

    return trained


# ─────────────────────────────────────────────
# SELECCIÓN
# ─────────────────────────────────────────────

def train_model_with_params(X_train, y_train, model_name, params, problem_type):
    """Entrena un modelo específico con hiperparámetros dados."""
    n_features = X_train.shape[1]

    pipelines = get_pipelines(n_features, problem_type, y_train, fast_mode=False)
    pipe = pipelines[model_name]

    # Aplicar parámetros
    params_prefixed = {f"model__{k}": v for k, v in params.items()}
    pipe.set_params(**params_prefixed)

    # Entrenar
    pipe.fit(X_train, y_train)

    return {model_name: pipe}


def get_best_model(cv_results, trained_models):

    metric = "f1_mean" if "f1_mean" in cv_results.columns else "r2_mean"

    best_row = cv_results.sort_values(by=metric, ascending=False).iloc[0]

    return best_row["model"], trained_models[best_row["model"]]


# ─────────────────────────────────────────────
# EVALUACIÓN FINAL
# ─────────────────────────────────────────────

def evaluate_trained_models(trained_models, X_test, y_test, problem_type):

    results = []

    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test)

            if problem_type == "classification":
                results.append({
                    "model": name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                })
            else:
                results.append({
                    "model": name,
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                })

        except Exception as e:
            logger.warning(f"{name} falló evaluación: {e}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# REPORTES
# ─────────────────────────────────────────────

def get_classification_report(best_model, X_test, y_test, label_encoder=None):

    y_pred = best_model.predict(X_test)

    target_names = (
        [str(c) for c in label_encoder.classes_]
        if label_encoder is not None
        else None
    )

    return classification_report(y_test, y_pred, target_names=target_names, zero_division=0)