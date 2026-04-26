# run_analysis.py

import os
import sys
import json
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from pipeline.data_loader import load_data
from pipeline.preprocessing import (
    split_data,
    detect_column_types,
    get_preprocessing_report,
    build_preprocessor
)
from pipeline.modeling import (
    evaluate_models,
    evaluate_models_robust,
    train_models,
    evaluate_trained_models,
    train_model_with_params
)
from pipeline.optimization import optimize_model
from pipeline.explainability import (
    compute_shap_values,
    compute_lime_explanation,
    generate_lime_text_explanation,
    get_shap_feature_importance,
)
from pipeline.utils import detect_target, detect_problem_type
from pipeline.reporting import generate_pdf_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs"


def main(file_path: str, use_optuna: bool = False, n_trials: int = 20) -> dict:

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    task_id = os.path.basename(file_path).split("_")[0]
    task_output_dir = os.path.join(OUTPUT_DIR, "tasks_output", task_id)
    os.makedirs(task_output_dir, exist_ok=True)

    start_time = datetime.now()
    best_params = None

    logger.info("=" * 60)
    logger.info("AGENTE DE ANÁLISIS AUTOMÁTICO — INICIO")
    logger.info(f"Archivo:  {file_path}")
    logger.info(f"Optuna:   {use_optuna} | Trials: {n_trials}")
    logger.info("=" * 60)

    results = {}

    try:
        # ── 1. Carga
        logger.info("\n[PASO 1] Cargando datos...")
        df = load_data(file_path)
        logger.info(f"Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")

        if df.shape[0] < 50:
            raise ValueError("Dataset demasiado pequeño")

        # ── 2. Target
        logger.info("\n[PASO 2] Detectando variable objetivo...")
        target = detect_target(df)
        logger.info(f"Target: '{target}'")

        problem_type = detect_problem_type(df[target])
        logger.info(f"Tipo de problema detectado: {problem_type}")

        if df[target].nunique() < 2:
            raise ValueError("Target sin variabilidad")

        # ── 3. Split
        logger.info("\n[PASO 3] Dividiendo datos...")
        X_train, X_test, y_train, y_test = split_data(
            df, target, problem_type
        )

        # ── 3b. Preprocesamiento
        logger.info("[PASO 3b] Construyendo preprocesador...")

        column_types = detect_column_types(df, target)
        preprocessor = build_preprocessor(column_types)

        preprocessing_report = get_preprocessing_report(column_types)

        # ── 4. Cross Validation
        logger.info("\n[PASO 4] Evaluando modelos...")
        cv_results = evaluate_models(
            X_train, y_train, problem_type, preprocessor
        )

        logger.info(f"\nRanking CV:\n{cv_results.to_string(index=False)}")

        # ── 5. Entrenamiento base
        selected_models = cv_results["model"].tolist()

        trained_models_all = train_models(
            X_train, y_train, selected_models, problem_type, preprocessor
        )

        # ── 6. Selección robusta
        logger.info("\n[PASO 6] Selección robusta...")

        robust_results = evaluate_models_robust(
            X_train, y_train, selected_models, problem_type, preprocessor
        )

        best_name = robust_results.iloc[0]["model"]
        logger.info(f"Mejor modelo: {best_name}")

        # ── 6b. Modelo final
        final_model = train_models(
            X_train, y_train, [best_name], problem_type, preprocessor
        )[best_name]

        # ── 6c. Optuna
        if use_optuna:
            logger.info("\n[PASO 6c] Optuna...")

            try:
                best_params = optimize_model(
                    best_name,
                    X_train,
                    y_train,
                    preprocessor,
                    problem_type,
                    n_trials=n_trials,
                    timeout=600
                )

                if best_params:

                    tuned_model = train_model_with_params(
                        X_train, y_train,
                        best_name,
                        best_params,
                        problem_type,
                        preprocessor
                    )[best_name]

                    if problem_type == "classification":
                        cv = StratifiedKFold(3, shuffle=True, random_state=42)
                        scoring = "f1_weighted"
                    else:
                        cv = KFold(3, shuffle=True, random_state=42)
                        scoring = "r2"

                    baseline = cross_val_score(
                        clone(final_model),
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=1
                    ).mean()

                    tuned = cross_val_score(
                        clone(tuned_model),
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=1
                    ).mean()

                    logger.info(f"Baseline: {baseline:.4f}")
                    logger.info(f"Tuned:    {tuned:.4f}")

                    if tuned > baseline:
                        final_model = tuned_model
                        trained_models_all[best_name] = tuned_model
                        logger.info("✔ Modelo optimizado seleccionado")

            except Exception as e:
                logger.warning(f"Optuna falló: {e}")

        # ── 7. Evaluación test
        logger.info("\n[PASO 7] Evaluando en test...")

        test_metrics = evaluate_trained_models(
            trained_models_all,
            X_test,
            y_test,
            problem_type
        )

        # Marcar el modelo seleccionado con ★ en lugar de agregar fila duplicada
        test_metrics["model"] = test_metrics["model"].apply(
            lambda m: f"★ {m}" if m == best_name else m
        )

        # ── 8. Explainability
        logger.info("\n[PASO 8] Explicabilidad...")

        # SHAP (seguro)
        try:
            shap_result = compute_shap_values(
                pipeline=final_model,
                X_train=X_train,
                y_train=y_train,
                sample_size=50
            )
            shap_importance = get_shap_feature_importance(shap_result)
        except Exception as e:
            logger.warning(f"SHAP falló: {e}")
            shap_result = {"error": str(e)}
            shap_importance = []

        # LIME (seguro)
        try:
            X_sample = X_train.sample(min(1000, len(X_train)), random_state=42)
            y_sample = y_train.loc[X_sample.index] if hasattr(y_train, "loc") else y_train

            lime_result = compute_lime_explanation(
                pipeline=final_model,
                X_train=X_sample,
                y_train=y_sample,
            )

            lime_data = generate_lime_text_explanation(lime_result)

            lime_text = lime_data["text"] if lime_data else "No disponible"
            lime_probs = lime_data.get("probabilities") if lime_data else None

        except Exception as e:
            logger.warning(f"LIME falló: {e}")
            lime_text = "No disponible"
            lime_probs = None

        # ── 9. PDF
        pdf_path = os.path.join(task_output_dir, "report.pdf")

        generate_pdf_report(
            output_path=pdf_path,
            target=target,
            best_model=best_name,
            metrics_df=test_metrics,
            lime_text=lime_text,          # ya es string, sin .get()
            problem_type=problem_type,
            shap_importance=shap_importance,
            preprocessing=preprocessing_report,
            elapsed_seconds=(datetime.now() - start_time).total_seconds(),
            probabilities=lime_probs,     # ya está extraído correctamente arriba
        )

        # ── 10. Resultado
        elapsed = (datetime.now() - start_time).total_seconds()

        results = {
            "status": "success",
            "run_info": {
                "file": os.path.basename(file_path),
                "target": target,
                "problem_type": problem_type,
                "best_model": best_name,
                "elapsed_seconds": round(elapsed, 2),
            },
            "preprocessing": preprocessing_report,
            "model_performance": {
                "cv_results": cv_results.to_dict(orient="records"),
                "test_metrics": test_metrics.to_dict(orient="records"),
            },
            "explainability": {
                "lime": {
                    "text": lime_text,
                    "probabilities": lime_probs
                },
                "shap": {
                    "feature_importance": shap_importance,
                    "status": "ok" if "error" not in shap_result else shap_result["error"]
                },
            },
            "optimization": {
                "enabled": use_optuna,
                "best_params": best_params,
            },
            "output_files": {
                "pdf": pdf_path,
                "json": os.path.join(task_output_dir, "results.json"),
            },
        }

    except Exception as e:
        logger.error(traceback.format_exc())

        results = {
            "status": "error",
            "error": str(e),
        }

    json_path = os.path.join(task_output_dir, "results.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=str)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python run_analysis.py <archivo.csv> [--optuna]")
        sys.exit(1)

    use_optuna = "--optuna" in sys.argv
    main(sys.argv[1], use_optuna)