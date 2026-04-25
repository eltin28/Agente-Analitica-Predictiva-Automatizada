# pipeline/preprocessing.py
"""
Preprocesamiento genérico para cualquier dataset tabular de clasificación.
No tiene hardcoding de nombres de columnas. Detecta automáticamente tipos,
cardinalidad, nulos y aplica transformaciones apropiadas.
"""

import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TRANSFORMERS PERSONALIZADOS
# ─────────────────────────────────────────────

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Elimina columnas numéricas con correlación de Pearson > threshold
    para reducir multicolinealidad antes del modelado.
    """
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).select_dtypes(include=[np.number])
        if X_df.shape[1] < 2:
            return self
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.columns_to_drop_ = [
            c for c in upper.columns if any(upper[c] > self.threshold)
        ]
        logger.info(f"CorrelationFilter: eliminando columnas {self.columns_to_drop_}")
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.drop(columns=self.columns_to_drop_, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([], dtype=object)
        return np.array(
            [f for f in input_features if f not in self.columns_to_drop_],
            dtype=object
        )


class WinsorizationTransformer(BaseEstimator, TransformerMixin):
    """
    Winsorización percentil: recorta valores extremos para reducir impacto de outliers.
    No agrega ni elimina columnas, solo recorta valores.
    """
    def __init__(self, lower_bound=0.01, upper_bound=0.99):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.limits_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.limits_ = {
            col: (
                X_df[col].quantile(self.lower_bound),
                X_df[col].quantile(self.upper_bound)
            )
            for col in X_df.columns
        }
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.limits_:
                lower, upper = self.limits_[col]
                X_df[col] = np.clip(X_df[col], lower, upper)
        return X_df.values

    def get_feature_names_out(self, input_features=None):
        """
        Requerido para que ColumnTransformer propague nombres de columnas.
        La winsorización no cambia cantidad ni orden de columnas.
        """
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return np.array(list(self.limits_.keys()), dtype=object)


class DropHighCardinalityTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas categóricas con cardinalidad > max_unique durante el fit.
    Útil para IDs, hashes, textos libres que no aportan al modelo.
    """
    def __init__(self, max_unique=50):
        self.max_unique = max_unique
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.columns_to_drop_ = [
            col for col in X_df.columns
            if X_df[col].nunique() > self.max_unique
        ]
        logger.info(
            f"DropHighCardinality: eliminando {len(self.columns_to_drop_)} "
            f"columnas de alta cardinalidad"
        )
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.drop(columns=self.columns_to_drop_, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([], dtype=object)
        return np.array(
            [f for f in input_features if f not in self.columns_to_drop_],
            dtype=object
        )


# ─────────────────────────────────────────────
# DETECCIÓN AUTOMÁTICA DE TIPOS DE COLUMNA
# ─────────────────────────────────────────────

# Umbrales configurables
MISSING_DROP_THRESHOLD = 0.40      # Drop columna si > 40% nulos
HIGH_CARDINALITY_THRESHOLD = 50    # Drop categórica si > 50 valores únicos
ORDINAL_UNIQUE_THRESHOLD = 10      # Tratar como ordinal si <= 10 valores únicos
ITERATIVE_IMPUTER_MAX_COLS = 30    # Usar IterativeImputer solo si hay <= 30 columnas numéricas
WINSORIZE = True                   # Aplicar winsorización a numéricas


def detect_column_types(df: pd.DataFrame, target: str) -> dict:
    """
    Analiza el DataFrame e infiere automáticamente qué tipo de
    transformación corresponde a cada columna.

    Returns:
        dict con claves:
            - numeric: columnas numéricas continuas
            - categorical_nominal: categóricas de baja cardinalidad (OneHot)
            - categorical_ordinal: categóricas de muy baja cardinalidad (Ordinal)
            - high_cardinality: categóricas con demasiados valores (drop)
            - drop_missing: columnas con > MISSING_DROP_THRESHOLD nulos
            - id_like: columnas que parecen identificadores (drop)
    """
    feature_cols = [c for c in df.columns if c != target]
    result = {
        "numeric": [],
        "categorical_nominal": [],
        "categorical_ordinal": [],
        "high_cardinality": [],
        "drop_missing": [],
        "id_like": [],
    }

    for col in feature_cols:
        series = df[col]

        # 1. Alta tasa de nulos → drop
        missing_ratio = series.isnull().mean()
        if missing_ratio > MISSING_DROP_THRESHOLD:
            result["drop_missing"].append(col)
            logger.info(f"  [{col}] → DROP_MISSING ({missing_ratio:.1%} nulos)")
            continue

        # 2. Columnas tipo ID: muchos únicos, sin patrones
        if _is_id_column(series):
            result["id_like"].append(col)
            logger.info(f"  [{col}] → ID_LIKE (posible identificador, se descarta)")
            continue

        dtype = series.dtype
        n_unique = series.nunique()

        # 3. Numéricas
        if pd.api.types.is_numeric_dtype(dtype):
            result["numeric"].append(col)
            logger.info(f"  [{col}] → NUMERIC ({n_unique} únicos)")

        # 4. Categóricas
        else:
            if n_unique > HIGH_CARDINALITY_THRESHOLD:
                result["high_cardinality"].append(col)
                logger.info(f"  [{col}] → HIGH_CARDINALITY ({n_unique} únicos, se descarta)")
            elif n_unique <= ORDINAL_UNIQUE_THRESHOLD:
                result["categorical_ordinal"].append(col)
                logger.info(f"  [{col}] → CATEGORICAL_ORDINAL ({n_unique} únicos)")
            else:
                result["categorical_nominal"].append(col)
                logger.info(f"  [{col}] → CATEGORICAL_NOMINAL ({n_unique} únicos)")

    return result


def _is_id_column(series: pd.Series) -> bool:
    """
    Heurística para detectar columnas tipo ID o clave primaria.
    Criterios:
      - Nombre sugiere ID (contiene 'id', 'key', 'code', 'uuid', 'hash')
      - Ratio únicos/filas > 0.95 y tipo string/int
    """
    name_lower = series.name.lower() if series.name else ""
    id_keywords = {"id", "key", "code", "uuid", "hash", "pk", "index"}

    # Nombre sugerente
    name_match = any(kw in name_lower for kw in id_keywords)

    # Casi todas las filas son únicas
    uniqueness_ratio = series.nunique() / max(len(series), 1)
    high_uniqueness = uniqueness_ratio > 0.95 and len(series) > 50

    return name_match or high_uniqueness


# ─────────────────────────────────────────────
# CONSTRUCCIÓN DINÁMICA DEL PREPROCESSOR
# ─────────────────────────────────────────────

def build_preprocessor(column_types: dict) -> ColumnTransformer:
    """
    Construye un ColumnTransformer dinámico basado en los tipos detectados.
    Cada grupo de columnas recibe su pipeline de transformación apropiada.
    """
    transformers = []

    # ── Numéricas ──────────────────────────────────────
    numeric_cols = column_types["numeric"]
    if numeric_cols:
        if len(numeric_cols) <= ITERATIVE_IMPUTER_MAX_COLS:
            imputer = IterativeImputer(random_state=42, max_iter=10)
        else:
            imputer = SimpleImputer(strategy="median")

        steps = [
            ("imputer", imputer),
            ("scaler", StandardScaler()),
        ]
        if WINSORIZE:
            # Winsorización va ANTES del scaler
            steps = [
                ("imputer", imputer),
                ("winsor", WinsorizationTransformer()),
                ("scaler", StandardScaler()),
            ]

        numeric_pipeline = Pipeline(steps)
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
        logger.info(f"Preprocessor: {len(numeric_cols)} columnas numéricas")

    # ── Categóricas nominales (OneHot) ─────────────────
    nominal_cols = column_types["categorical_nominal"]
    if nominal_cols:
        nominal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            )),
        ])
        transformers.append(("nominal", nominal_pipeline, nominal_cols))
        logger.info(f"Preprocessor: {len(nominal_cols)} columnas nominales (OneHot)")

    # ── Categóricas ordinales (Ordinal) ────────────────
    ordinal_cols = column_types["categorical_ordinal"]
    if ordinal_cols:
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))
        logger.info(f"Preprocessor: {len(ordinal_cols)} columnas ordinales")

    if not transformers:
        raise ValueError(
            "No se encontraron columnas válidas para procesar. "
            "Revisa el dataset o ajusta los umbrales de filtrado."
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ─────────────────────────────────────────────
# SPLIT ESTRATIFICADO
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame, target: str, problem_type: str = "classification"):
    """
    Separa el DataFrame en X/y y aplica train/test split estratificado.

    Args:
        df: DataFrame completo con features y target.
        target: nombre de la columna objetivo.
        problem_type: "classification" o "regression".

    Returns:
        X_train, X_test, y_train, y_test (sin procesar, listos para el preprocessor)
    """
    if target not in df.columns:
        raise ValueError(f"La columna target '{target}' no existe en el DataFrame.")

    if df[target].isnull().any():
        n_missing = df[target].isnull().sum()
        logger.warning(f"Target tiene {n_missing} nulos. Se eliminan esas filas.")
        df = df.dropna(subset=[target]).copy()

    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    logger.info(
        f"split_data: {len(X_train)} train / {len(X_test)} test "
        f"(stratify={stratify is not None})"
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# PREPARACIÓN DE TARGET
# ─────────────────────────────────────────────

def encode_target(y_train: pd.Series, y_test: pd.Series):
    """
    Codifica el target a enteros [0, n_classes-1].
    Retorna y_train_enc, y_test_enc, label_encoder.
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    logger.info(f"Target '{y_train.name}': {len(le.classes_)} clases → {list(le.classes_)}")
    return y_train_enc, y_test_enc, le


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame, target: str) -> tuple:
    """
    Pipeline completo de preprocesamiento genérico:

    1. Detecta tipos de columna automáticamente
    2. Separa features y target
    3. Hace train/test split estratificado
    4. Construye ColumnTransformer dinámico
    5. Aplica transformaciones
    6. Codifica el target
    7. Retorna DataFrames listos para el modelado

    Args:
        df: DataFrame con datos crudos
        target: nombre de la columna objetivo

    Returns:
        X_train, X_test, y_train, y_test, preprocessor, label_encoder, column_types
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PREPROCESAMIENTO GENÉRICO")
    logger.info(f"Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
    logger.info(f"Target: '{target}'")
    logger.info("=" * 60)

    # ── 1. Validación básica ──────────────────────────
    if target not in df.columns:
        raise ValueError(f"La columna target '{target}' no existe en el DataFrame.")

    if df[target].isnull().any():
        n_missing = df[target].isnull().sum()
        logger.warning(f"Target tiene {n_missing} nulos. Se eliminan esas filas.")
        df = df.dropna(subset=[target]).copy()

    # ── 2. Separar X e y ─────────────────────────────
    X = df.drop(columns=[target])
    y = df[target]

    # ── 3. Detección de tipos ─────────────────────────
    logger.info("\nDetectando tipos de columna:")
    column_types = detect_column_types(df, target)

    # Resumen de detección
    useful_cols = (
        column_types["numeric"]
        + column_types["categorical_nominal"]
        + column_types["categorical_ordinal"]
    )

    logger.info(f"\nResumen:")
    logger.info(f"  Numéricas:          {len(column_types['numeric'])}")
    logger.info(f"  Nominales (OneHot): {len(column_types['categorical_nominal'])}")
    logger.info(f"  Ordinales:          {len(column_types['categorical_ordinal'])}")
    logger.info(f"  Descartadas (nulos):{len(column_types['drop_missing'])}")
    logger.info(f"  Descartadas (card): {len(column_types['high_cardinality'])}")
    logger.info(f"  Descartadas (ID):   {len(column_types['id_like'])}")
    logger.info(f"  Total usables:      {len(useful_cols)}")

    if len(useful_cols) == 0:
        raise ValueError(
            "El dataset no tiene columnas utilizables después del filtrado. "
            "Ajusta los umbrales en preprocessing.py."
        )

    # ── 4. Split estratificado ────────────────────────
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info(f"\nSplit: {len(X_train_raw)} train / {len(X_test_raw)} test")

    # ── 5. Construir y aplicar preprocessor ──────────
    preprocessor = build_preprocessor(column_types)

    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # ── 6. Nombres de features ────────────────────────
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [
            name.split("__", 1)[-1] if "__" in name else name
            for name in feature_names
        ]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    X_train_final = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_final = pd.DataFrame(X_test_processed, columns=feature_names)

    # ── 7. Codificar target ───────────────────────────
    y_train_enc, y_test_enc, label_encoder = encode_target(y_train_raw, y_test_raw)

    y_train_final = pd.Series(y_train_enc, name=target)
    y_test_final = pd.Series(y_test_enc, name=target)

    logger.info(f"\nShape final → X_train: {X_train_final.shape}, X_test: {X_test_final.shape}")
    logger.info("PREPROCESAMIENTO COMPLETADO\n")

    return (
        X_train_final,
        X_test_final,
        y_train_final,
        y_test_final,
        preprocessor,
        label_encoder,
        column_types,
    )


# ─────────────────────────────────────────────
# UTILIDADES DE INSPECCIÓN
# ─────────────────────────────────────────────

def get_preprocessing_report(column_types: dict) -> dict:
    """
    Genera un resumen del preprocesamiento para incluir en el JSON de resultados.
    """
    return {
        "numeric_features": column_types["numeric"],
        "nominal_features": column_types["categorical_nominal"],
        "ordinal_features": column_types["categorical_ordinal"],
        "dropped_high_missing": column_types["drop_missing"],
        "dropped_high_cardinality": column_types["high_cardinality"],
        "dropped_id_like": column_types["id_like"],
        "total_input_features": sum(len(v) for v in column_types.values()),
        "total_used_features": (
            len(column_types["numeric"])
            + len(column_types["categorical_nominal"])
            + len(column_types["categorical_ordinal"])
        ),
    }