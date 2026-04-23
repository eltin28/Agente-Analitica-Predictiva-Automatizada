import numpy as np
import pandas as pd
import logging

import shap
import lime.lime_tabular
from pipeline.utils import detect_problem_type

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG (CLAVE PARA PERFORMANCE)
# ─────────────────────────────────────────────

SHAP_SAMPLE_SIZE = 100
SHAP_BACKGROUND_SIZE = 50
LIME_SAMPLE_SIZE = 1000  # limitar dataset para LIME

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def _get_final_estimator(pipeline):
    return pipeline.named_steps.get("model", pipeline)


def _is_tree_based(estimator) -> bool:
    name = type(estimator).__name__.lower()
    return any(x in name for x in ["tree", "forest", "lgbm", "xgb"])


def _transform_without_model(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta SOLO preprocessing (sin modelo).
    Mucho más eficiente que recomputar múltiples veces.
    """
    X_out = X.copy()

    for name, step in pipeline.named_steps.items():
        if name == "model":
            break
        if name == "smote":
            continue
        if hasattr(step, "transform"):
            try:
                X_out = step.transform(X_out)
            except Exception as e:
                logger.warning(f"Transform falló en {name}: {e}")

    if isinstance(X_out, np.ndarray):
        return pd.DataFrame(X_out)

    return X_out


# ─────────────────────────────────────────────
# SHAP (OPTIMIZADO)
# ─────────────────────────────────────────────

def compute_shap_values(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series = None,
    label_encoder=None,
    sample_size: int = SHAP_SAMPLE_SIZE,
):
    """
    Versión optimizada:
    - Sampleo controlado
    - Evita KernelExplainer en la medida de lo posible
    - Fallback seguro
    """

    problem_type = detect_problem_type(y_train) if y_train is not None else "regression"
    estimator = _get_final_estimator(pipeline)

    # ── Sampling (CRÍTICO)
    X_sample = X_train.sample(
        min(sample_size, len(X_train)),
        random_state=42
    )

    try:
        X_transformed = _transform_without_model(pipeline, X_sample)

        # ── Caso rápido (modelos árbol)
        if _is_tree_based(estimator):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_transformed)

            return {
                "shap_values": shap_values,
                "feature_names": list(X_transformed.columns),
                "explainer_type": "TreeExplainer",
                "problem_type": problem_type,
            }

        # ── Caso no soportado → degradar a importance básica
        logger.warning("SHAP no disponible para este modelo")

        return {
            "shap_values": None,
            "feature_names": list(X_transformed.columns),
            "explainer_type": "NotAvailable",
            "problem_type": problem_type,
        }

    except Exception as e:
        logger.error(f"SHAP falló: {e}")
        return {"error": str(e), "shap_values": None}


# ─────────────────────────────────────────────
# LIME (OPTIMIZADO)
# ─────────────────────────────────────────────

def compute_lime_explanation(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series = None,
    label_encoder=None,
    instance_index: int = 0,
    num_features: int = 8,
):
    """
    Optimizado:
    - Limita dataset de entrenamiento de LIME
    - Evita usar todo X_train (crítico)
    """

    problem_type = detect_problem_type(y_train) if y_train is not None else "regression"

    # ── Sampleo (CRÍTICO)
    X_sample = X_train.sample(
        min(LIME_SAMPLE_SIZE, len(X_train)),
        random_state=42
    )

    if instance_index >= len(X_sample):
        instance_index = 0

    feature_names = X_sample.columns.tolist()
    instance = X_sample.iloc[instance_index]

    mode = "classification" if problem_type == "classification" else "regression"

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_sample.values,
        feature_names=feature_names,
        mode=mode,
        random_state=42,
    )

    try:
        if problem_type == "classification":

            class_names = (
                [str(c) for c in label_encoder.classes_]
                if label_encoder else None
            )

            exp = explainer.explain_instance(
                instance.values,
                pipeline.predict_proba,
                num_features=num_features,
                num_samples=500
            )

            instance_df = pd.DataFrame([instance.values], columns=feature_names)
            proba = pipeline.predict_proba(instance_df)[0]
            pred_class = int(np.argmax(proba))

            return {
                "type": "classification",
                "predicted_class": (
                    class_names[pred_class] if class_names else pred_class
                ),
                "probabilities": {
                    (class_names[i] if class_names else i): float(p)
                    for i, p in enumerate(proba)
                },
                "explanation": exp.as_list(),
            }

        else:
            exp = explainer.explain_instance(
                instance.values,
                pipeline.predict,
                num_features=num_features,
                num_samples=500
            )

            pred = pipeline.predict(instance.values.reshape(1, -1))[0]

            return {
                "type": "regression",
                "prediction": float(pred),
                "explanation": exp.as_list(),
            }

    except Exception as e:
        logger.error(f"LIME falló: {e}")
        return {"error": str(e), "explanation": []}


# ─────────────────────────────────────────────
# TEXTO LIME
# ─────────────────────────────────────────────

def generate_lime_text_explanation(lime_result: dict) -> str:

    if "error" in lime_result:
        return f"No se pudo generar explicación: {lime_result['error']}"

    lines = []

    if lime_result.get("type") == "classification":
        lines.append(f"Clase predicha: {lime_result.get('predicted_class')}")
        lines.append("")

        probs = lime_result.get("probabilities", {})
        if probs:
            lines.append("Probabilidades:")
            for k, v in probs.items():
                lines.append(f"{k}: {v:.2%}")
            lines.append("")
    else:
        lines.append(f"Valor predicho: {lime_result.get('prediction')}")
        lines.append("")

    for feature, weight in lime_result.get("explanation", []):
        direction = "aumenta" if weight > 0 else "disminuye"
        lines.append(f"{feature} → {direction} ({round(weight,4)})")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# SHAP IMPORTANCE
# ─────────────────────────────────────────────

def get_shap_feature_importance(shap_result: dict):

    shap_values = shap_result.get("shap_values")
    feature_names = shap_result.get("feature_names", [])

    if shap_values is None:
        return []

    try:
        if isinstance(shap_values, list):
            importance = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values],
                axis=0
            )
        else:
            importance = np.abs(shap_values).mean(axis=0)

        idx = np.argsort(importance)[::-1]

        return [
            {"feature": feature_names[i], "importance": float(importance[i])}
            for i in idx
        ]

    except Exception as e:
        logger.warning(f"SHAP importance falló: {e}")
        return []