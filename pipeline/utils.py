# pipeline/utils.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DETECCIÓN DE TARGET (ROBUSTA)
# ─────────────────────────────────────────────

def detect_target(df: pd.DataFrame) -> str:
    """
    Detecta la variable objetivo usando heurísticas robustas.

    Estrategia:
    1. Prioridad por nombre
    2. Columnas categóricas con baja cardinalidad
    3. Columnas binarias
    4. Fallback: última columna
    """

    priority_names = {
        "target", "label", "class", "outcome",
        "survived", "default", "y"
    }

    cols = df.columns.tolist()

    # ── 1. Match por nombre (case-insensitive)
    for col in cols:
        if col.lower() in priority_names:
            logger.info(f"Target detectado por nombre: {col}")
            return col

    # ── 2. Candidatos categóricos (baja cardinalidad)
    candidates = []

    for col in cols:
        n_unique = df[col].nunique()

        # Evitar columnas casi constantes
        if n_unique <= 1:
            continue

        # Categórico o pocos valores
        if n_unique <= 10:
            candidates.append((col, n_unique))

    if candidates:
        # Elegir el más "informativo" (más clases)
        selected = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
        logger.info(f"Target detectado por cardinalidad: {selected}")
        return selected

    # ── 3. Binarios (fallback)
    binary_cols = [col for col in cols if df[col].nunique() == 2]

    if binary_cols:
        selected = binary_cols[-1]
        logger.info(f"Target detectado como binario: {selected}")
        return selected

    # ── 4. Fallback final
    fallback = cols[-1]
    logger.warning(f"No se detectó target claro. Usando última columna: {fallback}")
    return fallback


# ─────────────────────────────────────────────
# DETECCIÓN DE PROBLEMA (UNIFICADA)
# ─────────────────────────────────────────────

def detect_problem_type(y: pd.Series) -> str:
    """
    Detecta si el problema es clasificación o regresión.

    Reglas consistentes con modeling:
    - Tipo object → clasificación
    - Pocos valores únicos → clasificación
    - Muchos valores únicos → regresión
    """

    if y.dtype == "object":
        return "classification"

    unique_values = y.nunique()
    total_values = len(y)

    if unique_values <= 20 or (unique_values / total_values) < 0.05:
        return "classification"

    return "regression"