# pipeline/optimization.py

import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from pipeline.modeling import get_model_instance


def _get_search_space(trial, model_name: str):
    """
    OCP: cada modelo define su propio espacio.
    """

    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }

    elif model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        }

    elif model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 0.1, 10),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
        }

    elif model_name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }

    elif model_name == "MLP":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(50,), (100,), (100, 50)]
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        }

    elif model_name == "DecisionTree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }

    return {}


def optimize_model(
    model_name: str,
    X,
    y,
    preprocessor,
    problem_type: str,
    n_trials: int = 20,
    timeout: int = 600,
):
    """
    Optimización robusta sobre pipeline completo.
    """

    def objective(trial):
        try:
            params = _get_search_space(trial, model_name)

            model = get_model_instance(model_name, problem_type)
            model.set_params(**params)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            if problem_type == "classification":
                cv = StratifiedKFold(3, shuffle=True, random_state=42)
                scoring = "f1_weighted"
            else:
                cv = KFold(3, shuffle=True, random_state=42)
                scoring = "r2"

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,  # obligatorio en tu arquitectura
                error_score="raise"  # detecta errores reales
            )

            return scores.mean()

        except Exception:
            # Evita que Optuna se rompa por un modelo malo
            return float("-inf")

    # ✔ reproducibilidad
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False
    )

    if study.best_trial is None:
        return None

    return study.best_params