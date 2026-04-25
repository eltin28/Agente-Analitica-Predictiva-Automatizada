import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DETECCIÓN DE TARGET (ROBUSTA)
# ─────────────────────────────────────────────

def detect_target(df: pd.DataFrame) -> str:
    priority_names = {
        "target", "label", "class", "outcome",
        "survived", "default", "y"
    }

    cols = df.columns.tolist()

    # ── 1. Match por nombre
    for col in cols:
        if col.lower() in priority_names:
            logger.info(f"Target detectado por nombre: {col}")
            return col

    # ── 2. Candidatos categóricos
    candidates = []

    for col in cols:
        n_unique = df[col].nunique()

        if n_unique <= 1:
            continue

        if n_unique <= 10:
            candidates.append((col, n_unique))

    if candidates:
        selected = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
        logger.info(f"Target detectado por cardinalidad: {selected}")
        return selected

    # ── 3. Binarios
    binary_cols = [col for col in cols if df[col].nunique() == 2]

    if binary_cols:
        selected = binary_cols[-1]
        logger.info(f"Target detectado como binario: {selected}")
        return selected

    # ── 4. Fallback
    fallback = cols[-1]
    logger.warning(f"No se detectó target claro. Usando última columna: {fallback}")
    return fallback


# ─────────────────────────────────────────────
# DETECCIÓN DE PROBLEMA (ROBUSTA Y CONSISTENTE)
# ─────────────────────────────────────────────

def detect_problem_type(y) -> str:
    """
    Detecta tipo de problema de forma robusta.
    Compatible con:
    - pandas.Series
    - numpy.ndarray
    """

    # Normalizar entrada
    if isinstance(y, np.ndarray):
        y_series = pd.Series(y)
    elif isinstance(y, pd.Series):
        y_series = y
    else:
        y_series = pd.Series(y)

    # ── Clasificación por tipo
    if y_series.dtype == "object":
        return "classification"

    # ── Cardinalidad
    unique_values = y_series.nunique()
    total_values = len(y_series)

    # Heurística más estable
    if unique_values <= 20:
        return "classification"

    if (unique_values / total_values) < 0.05:
        return "classification"

    return "regression"