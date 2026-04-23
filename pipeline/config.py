"""
Configuración central del pipeline de analítica.

Define reglas de negocio, modos de ejecución y parámetros globales.
Este archivo evita hardcoding y permite comportamiento adaptativo.
"""

from enum import Enum


# ─────────────────────────────────────────────
# MODOS DE EJECUCIÓN
# ─────────────────────────────────────────────

class ExecutionMode(str, Enum):
    FAST = "fast"          # datasets pequeños / respuesta rápida
    BALANCED = "balanced"  # equilibrio
    FULL = "full"          # máxima calidad (más lento)


# ─────────────────────────────────────────────
# DETECCIÓN DE TAMAÑO DE DATASET
# ─────────────────────────────────────────────

def detect_execution_mode(n_rows: int) -> ExecutionMode:
    """
    Determina el modo de ejecución según tamaño del dataset.
    """
    if n_rows < 5_000:
        return ExecutionMode.FAST

    if n_rows < 50_000:
        return ExecutionMode.BALANCED

    return ExecutionMode.FULL


# ─────────────────────────────────────────────
# CONFIGURACIÓN DE MODELADO
# ─────────────────────────────────────────────

MODEL_CONFIG = {

    ExecutionMode.FAST: {
        "models": ["DecisionTree", "KNN", "RandomForest", "LightGBM"],
        "cv_folds_fast": 3,
        "cv_folds_full": 3,
        "top_k_models": 2,
        "use_mlp": False,
        "use_svm": False,
    },

    ExecutionMode.BALANCED: {
        "models": ["DecisionTree", "KNN", "SVM", "RandomForest", "LightGBM"],
        "cv_folds_fast": 3,
        "cv_folds_full": 5,
        "top_k_models": 3,
        "use_mlp": False,
        "use_svm": True,
    },

    ExecutionMode.FULL: {
        "models": ["DecisionTree", "KNN", "SVM", "RandomForest", "LightGBM", "MLP"],
        "cv_folds_fast": 3,
        "cv_folds_full": 5,
        "top_k_models": 3,
        "use_mlp": True,
        "use_svm": True,
    },
}


# ─────────────────────────────────────────────
# SMOTE
# ─────────────────────────────────────────────

SMOTE_CONFIG = {
    "enabled": True,
    "imbalance_threshold": 0.20,  # clase minoritaria < 20%
    "apply_in_fast_phase": False, # CRÍTICO para rendimiento
}


# ─────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────

FEATURE_SELECTION_CONFIG = {
    "enabled": True,
    "ratio": 0.3,   # % de features a seleccionar
    "min_features": 5,
    "apply_to_models": ["SVM", "KNN", "MLP"],
}


# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────

SHAP_CONFIG = {
    ExecutionMode.FAST: {
        "sample_size": 50,
    },
    ExecutionMode.BALANCED: {
        "sample_size": 100,
    },
    ExecutionMode.FULL: {
        "sample_size": 200,
    },
}


# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────

LIME_CONFIG = {
    "num_features": 8
}