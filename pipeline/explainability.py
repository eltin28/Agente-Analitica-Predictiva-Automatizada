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
    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")

    if preprocessor is None or model is None:
        raise ValueError("Pipeline inválido: faltan 'preprocessor' o 'model'")

    return preprocessor, model


def _transform_data(preprocessor, X):
    """
    Aplica transformación y devuelve DataFrame 2D float con nombres limpios.

    Estrategia de nombres (en orden de prioridad):
      1. get_feature_names_out() del ColumnTransformer → limpiar prefijo 'grupo__'
      2. Fallback numérico 'feature_N' (nunca mezcla nombres originales con índices)
    """
    X_t = preprocessor.transform(X)

    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    X_t = np.array(X_t, dtype=float)

    try:
        raw_names = preprocessor.get_feature_names_out()
        feature_names = [
            name.split("__", 1)[-1] if "__" in name else name
            for name in raw_names
        ]
        if len(feature_names) != X_t.shape[1]:
            raise ValueError(
                f"Mismatch: {len(feature_names)} nombres vs {X_t.shape[1]} columnas"
            )
    except Exception as e:
        logger.warning(f"_transform_data: fallback a nombres genéricos ({e})")
        feature_names = [f"feature_{i}" for i in range(X_t.shape[1])]

    return pd.DataFrame(X_t, columns=feature_names, index=range(X_t.shape[0]))


# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────

def compute_shap_values(pipeline, X_train, y_train=None, sample_size=100):
    try:
        preprocessor, model = _split_pipeline(pipeline)

        X_sample = X_train.sample(min(sample_size, len(X_train)), random_state=42)
        X_transformed = _transform_data(preprocessor, X_sample)

        model_class = type(model).__name__
        is_tree = any(k in model_class for k in ("Forest", "Tree", "LGBM", "XGB", "Gradient"))

        if is_tree:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
        elif hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model, X_transformed)
            shap_values = explainer(X_transformed).values
        else:
            explainer = shap.Explainer(model.predict, X_transformed)
            shap_values = explainer(X_transformed).values

        # Normalizar a (n_samples, n_features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

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
    if "error" in shap_result:
        return []

    values = np.array(shap_result["values"], dtype=float)
    feature_names = shap_result["feature_names"]

    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim == 3:
        values = values[:, :, 1]

    importance = np.abs(values).mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    return df.to_dict(orient="records")


# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────

def compute_lime_explanation(pipeline, X_train, y_train, sample_index=0):
    try:
        preprocessor, model = _split_pipeline(pipeline)
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
            exp = explainer.explain_instance(instance, model.predict_proba, num_features=10)
            probs = model.predict_proba([instance])[0].tolist()
        else:
            exp = explainer.explain_instance(instance, model.predict, num_features=10)
            probs = None

        return {"exp": exp, "probabilities": probs}

    except Exception as e:
        logger.warning(f"LIME error: {e}")
        return None


def generate_lime_text_explanation(lime_result):
    if lime_result is None:
        return None

    try:
        exp = lime_result["exp"]
        lines = [f"{feature}: {weight:.4f}" for feature, weight in exp.as_list()]
        return {
            "text": "\n".join(lines),
            "probabilities": lime_result.get("probabilities")
        }

    except Exception as e:
        logger.warning(f"LIME text error: {e}")
        return None