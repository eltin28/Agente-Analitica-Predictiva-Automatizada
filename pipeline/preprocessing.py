# pipeline/preprocessing.py

import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from pipeline.utils import detect_problem_type

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG GLOBAL
# ─────────────────────────────────────────────

MISSING_DROP_THRESHOLD = 0.40
HIGH_CARDINALITY_THRESHOLD = 50
ORDINAL_UNIQUE_THRESHOLD = 10
WINSORIZE = True


# ─────────────────────────────────────────────
# TRANSFORMERS OPTIMIZADOS
# ─────────────────────────────────────────────

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)

        # OPTIMIZACIÓN: limitar a 100 features máximo
        if X_df.shape[1] > 100:
            logger.info("CorrelationFilter skip (demasiadas features)")
            return self

        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        self.columns_to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]

        return self

    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.columns_to_drop_, errors="ignore")


class WinsorizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.bounds_ = {
            c: (X[c].quantile(self.lower), X[c].quantile(self.upper))
            for c in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (l, u) in self.bounds_.items():
            X[col] = np.clip(X[col], l, u)
        return X

# ─────────────────────────────────────────────
# DETECCIÓN DE COLUMNAS
# ─────────────────────────────────────────────

def detect_column_types(df, target):

    result = {
        "numeric": [],
        "categorical_nominal": [],
        "categorical_ordinal": [],
        "drop": [],
    }

    for col in df.columns:
        if col == target:
            continue

        s = df[col]

        if s.isnull().mean() > MISSING_DROP_THRESHOLD:
            result["drop"].append(col)
            continue

        if _is_id_column(s):
            result["drop"].append(col)
            continue

        n_unique = s.nunique()

        if pd.api.types.is_numeric_dtype(s):
            result["numeric"].append(col)
        elif n_unique > HIGH_CARDINALITY_THRESHOLD:
            result["categorical_nominal"].append(col)  # pero tratar diferente
        elif n_unique <= ORDINAL_UNIQUE_THRESHOLD:
            result["categorical_ordinal"].append(col)
        else:
            result["categorical_nominal"].append(col)

    return result


def _is_id_column(series):
    ratio = series.nunique() / max(len(series), 1)
    return ratio > 0.95 and len(series) > 50


# ─────────────────────────────────────────────
# PREPROCESSOR OPTIMIZADO
# ─────────────────────────────────────────────

def build_preprocessor(column_types, fast_mode=False):

    transformers = []

    # ── NUMÉRICAS ─────────────────────────────
    if column_types["numeric"]:

        steps = [
            ("imputer", SimpleImputer(strategy="median")),
        ]

        # Winsorization opcional (no bloquear pipeline)
        if WINSORIZE and not fast_mode:
            steps.append(("winsor", WinsorizationTransformer()))

        steps.append(("scaler", StandardScaler()))

        transformers.append((
            "num",
            Pipeline(steps),
            column_types["numeric"]
        ))

    # ── NOMINALES ─────────────────────────────
    if column_types["categorical_nominal"]:
        transformers.append((
            "nom",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=50
                )),
            ]),
            column_types["categorical_nominal"]
        ))

    # ── ORDINALES ─────────────────────────────
    if column_types["categorical_ordinal"]:
        transformers.append((
            "ord",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )),
            ]),
            column_types["categorical_ordinal"]
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ─────────────────────────────────────────────
# EXTRACCIÓN DE NOMBRES DE FEATURES
# ─────────────────────────────────────────────

def _get_feature_names(preprocessor):
    """Extrae nombres de features después de ColumnTransformer."""
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "nom":
            if hasattr(transformer.named_steps["onehot"], "get_feature_names_out"):
                try:
                    names = transformer.named_steps["onehot"].get_feature_names_out(columns)
                    feature_names.extend(names)
                except:
                    feature_names.extend(columns)
            else:
                feature_names.extend(columns)
        elif name == "ord":
            feature_names.extend(columns)
    return feature_names if feature_names else [f"feature_{i}" for i in range(100)]


# ─────────────────────────────────────────────
# REPORTE DE PREPROCESAMIENTO
# ─────────────────────────────────────────────

def get_preprocessing_report(column_types: dict) -> dict:
    """Genera reporte de preprocesamiento."""
    return {
        "numeric_columns": len(column_types.get("numeric", [])),
        "categorical_nominal_columns": len(column_types.get("categorical_nominal", [])),
        "categorical_ordinal_columns": len(column_types.get("categorical_ordinal", [])),
        "dropped_columns": len(column_types.get("drop", [])),
        "columns_used": (
            column_types.get("numeric", [])
            + column_types.get("categorical_nominal", [])
            + column_types.get("categorical_ordinal", [])
        ),
        "columns_dropped": column_types.get("drop", []),
    }


# ─────────────────────────────────────────────
# TARGET
# ─────────────────────────────────────────────

def encode_target(y_train, y_test, problem_type):

    if problem_type == "classification":
        le = LabelEncoder()
        return le.fit_transform(y_train), le.transform(y_test), le

    return y_train.values, y_test.values, None


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def preprocess_data(df, target, problem_type=None, fast_mode=False):

    if problem_type is None:
        problem_type = detect_problem_type(df[target])

    if target not in df.columns:
        raise ValueError("Target no encontrado")

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    column_types = detect_column_types(df, target)

    stratify = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    preprocessor = build_preprocessor(column_types, fast_mode)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    feature_names = _get_feature_names(preprocessor)

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    y_train, y_test, le = encode_target(y_train, y_test, problem_type)

    return X_train, X_test, y_train, y_test, preprocessor, le, column_types