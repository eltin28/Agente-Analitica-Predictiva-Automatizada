# pipeline/explainability.py

import numpy as np
import pandas as pd
import logging

import shap
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def _split_pipeline(pipeline):
    """
    Separa preprocessor y modelo desde Pipeline.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")

    if preprocessor is None or model is None:
        raise ValueError("Pipeline inválido: faltan 'preprocessor' o 'model'")

    return preprocessor, model


def _transform_data(preprocessor, X):
    """
    Aplica transformación y devuelve DataFrame con nombres de columnas robustos.
    """

    # Transformación
    X_t = preprocessor.transform(X)

    # Convertir a array primero (evita problemas después)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    X_t = np.array(X_t, dtype=float)

    feature_names = None

    try:
        raw_names = preprocessor.get_feature_names_out()

        cleaned = []
        for name in raw_names:
            name = name.split("__", 1)[-1] if "__" in name else name
            cleaned.append(name)

        # Validación fuerte
        if any(n.startswith(("x", "f_")) for n in cleaned):
            raise ValueError("Feature names poco interpretables")

        feature_names = cleaned

    except Exception:
        input_cols = list(X.columns)

        # Caso ideal: misma cantidad
        if len(input_cols) == X_t.shape[1]:
            feature_names = input_cols

        else:
            # fallback controlado
            feature_names = [
                f"{input_cols[i % len(input_cols)]}_{i}"
                for i in range(X_t.shape[1])
            ]

    return pd.DataFrame(X_t, columns=feature_names)


# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────

def compute_shap_values(
    pipeline,
    X_train,
    y_train=None,
    sample_size=100
):
    """
    Calcula SHAP usando modelo dentro del pipeline.
    Maneja correctamente clasificación binaria, multiclase y regresión.
    """

    try:
        preprocessor, model = _split_pipeline(pipeline)

        X_sample = X_train.sample(
            min(sample_size, len(X_train)),
            random_state=42
        )

        X_transformed = _transform_data(preprocessor, X_sample)

        # Usar TreeExplainer para modelos de árbol (más estable que Explainer genérico)
        model_class = type(model).__name__
        is_tree = any(k in model_class for k in ("Forest", "Tree", "LGBM", "XGB", "Gradient"))

        if is_tree:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
        elif hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model, X_transformed)
            shap_obj = explainer(X_transformed)
            shap_values = shap_obj.values
        else:
            explainer = shap.Explainer(model.predict, X_transformed)
            shap_obj = explainer(X_transformed)
            shap_values = shap_obj.values

        # ── Normalizar a shape (n_samples, n_features) ──────────────────
        # TreeExplainer en clasificación binaria devuelve lista [neg, pos]
        # o array 3D (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            # Lista de arrays por clase → tomar clase positiva (índice 1)
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        elif shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) → tomar clase 1
            shap_values = shap_values[:, :, 1]

        # Asegurar 2D float
        shap_values = np.array(shap_values, dtype=float)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        return {
            "values": shap_values,
            "feature_names": list(X_transformed.columns),
        }

    except Exception as e:
        logger.warning(f"SHAP error: {e}")
        return {"error": str(e)}


def get_shap_feature_importance(shap_result):
    """
    Devuelve importancia global de features.
    Espera shap_result["values"] con shape (n_samples, n_features).
    """

    if "error" in shap_result:
        return []

    values = np.array(shap_result["values"], dtype=float)
    feature_names = shap_result["feature_names"]

    # Garantizar 2D
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim == 3:
        values = values[:, :, 1]

    importance = np.abs(values).mean(axis=0)  # shape: (n_features,)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    return df.to_dict(orient="records")


# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────

def compute_lime_explanation(
    pipeline,
    X_train,
    y_train,
    sample_index=0
):
    """
    LIME sobre datos transformados (numéricos) pero manteniendo nombres interpretables.
    """

    try:
        preprocessor, model = _split_pipeline(pipeline)

        # Transformación segura (todo numérico)
        X_transformed = _transform_data(preprocessor, X_train)

        is_classification = hasattr(model, "predict_proba")

        explainer = LimeTabularExplainer(
            training_data=X_transformed.values,
            feature_names=X_transformed.columns.tolist(),
            mode="classification" if is_classification else "regression",
            discretize_continuous=True,
        )

        instance = X_transformed.iloc[sample_index].values

        if is_classification:
            exp = explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features=10
            )
            probs = model.predict_proba([instance])[0].tolist()
        else:
            exp = explainer.explain_instance(
                instance,
                model.predict,
                num_features=10
            )
            probs = None

        return {
            "exp": exp,
            "probabilities": probs
        }

    except Exception as e:
        logger.warning(f"LIME error: {e}")
        return None


def generate_lime_text_explanation(lime_result):
    if lime_result is None:
        return None

    try:
        exp = lime_result["exp"]
        explanation = exp.as_list()
        lines = [f"{feature}: {weight:.4f}" for feature, weight in explanation]

        return {
            "text": "\n".join(lines),
            "probabilities": lime_result.get("probabilities")
        }

    except Exception as e:
        logger.warning(f"LIME text error: {e}")
        return None