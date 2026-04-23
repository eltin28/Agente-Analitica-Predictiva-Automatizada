# run_analysis.py
"""
Pipeline CRISP-DM completo con:
- Preprocesamiento dinámico y genérico
- Evaluación multi-modelo con cross validation
- Optimización opcional con Optuna
- Explicabilidad (SHAP + LIME)
- Reporte PDF + JSON estructurado
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime

from pipeline.data_loader import load_data
from pipeline.preprocessing import preprocess_data, get_preprocessing_report
from pipeline.modeling import (
    evaluate_models,
    train_models,
    get_best_model,
    evaluate_trained_models,
    get_classification_report,
    detect_problem_type,
    train_model_with_params,
)
from pipeline.optimization import optimize_model
from pipeline.explainability import (
    compute_shap_values,
    compute_lime_explanation,
    generate_lime_text_explanation,
    get_shap_feature_importance,
)
from pipeline.utils import detect_target
from pipeline.reporting import generate_pdf_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs"


def main(file_path: str, use_optuna: bool = False, n_trials: int = 20) -> dict:
    """
    Ejecuta el pipeline CRISP-DM completo.

    Args:
        file_path:   ruta al CSV o Excel
        use_optuna:  activa optimización de hiperparámetros con Optuna
        n_trials:    número de trials de Optuna (solo si use_optuna=True)

    Returns:
        dict con todos los resultados, serializable a JSON.
        Estructura:
            status          → "success" | "error"
            run_info        → metadata del análisis
            preprocessing   → columnas usadas / descartadas
            model_performance → cv_results, test_metrics, classification_report
            explainability  → LIME text + SHAP feature importance
            optimization    → params encontrados por Optuna (si aplica)
            output_files    → rutas al PDF y JSON
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # El PDF y JSON se guardan en un subdirectorio por task_id
    # para que la API pueda servirlos por /download/{task_id}
    task_id = os.path.basename(file_path).split("_")[0]
    task_output_dir = os.path.join(OUTPUT_DIR, "tasks_output", task_id)
    os.makedirs(task_output_dir, exist_ok=True)

    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("AGENTE DE ANÁLISIS AUTOMÁTICO — INICIO")
    logger.info(f"Archivo:  {file_path}")
    logger.info(f"Optuna:   {use_optuna} | Trials: {n_trials}")
    logger.info("=" * 60)

    results = {}

    try:
        # ── 1. Carga ──────────────────────────────────
        logger.info("\n[PASO 1] Cargando datos...")
        df = load_data(file_path)
        logger.info(f"Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")

        # ── 2. Target ─────────────────────────────────
        logger.info("\n[PASO 2] Detectando variable objetivo...")
        target = detect_target(df)
        logger.info(f"Target: '{target}'")


        problem_type = detect_problem_type(df[target])
        logger.info(f"Tipo de problema detectado: {problem_type}")


        # ── 3. Preprocesamiento ───────────────────────
        logger.info("\n[PASO 3] Preprocesando datos...")
        (
            X_train, X_test,
            y_train, y_test,
            preprocessor,
            label_encoder,
            column_types,
        ) = preprocess_data(df, target)

        preprocessing_report = get_preprocessing_report(column_types)

        # ── 4. Cross Validation ───────────────────────
        logger.info("\n[PASO 4] Evaluando modelos (cross validation)...")
        cv_results = evaluate_models(X_train, y_train)
        logger.info(f"\nRanking CV:\n{cv_results.to_string(index=False)}")

        # ── 5. Entrenamiento base ─────────────────────
        logger.info("\n[PASO 5] Entrenando modelos base...")
        trained_models = train_models(X_train, y_train)

       # ── 6. Selección del mejor modelo ─────────────
        logger.info("\n[PASO 6] Seleccionando mejor modelo...")
        best_name, best_model = get_best_model(cv_results, trained_models)
        logger.info(f"Mejor modelo base: {best_name}")

        # Guardar referencia del modelo base
        final_model = best_model
        best_params = None

        # ── 6b. Optimización con Optuna (opcional) ────
        if use_optuna:
            logger.info("\n[PASO 6b] Optimizando hiperparámetros con Optuna...")
            try:
                best_params = optimize_model(
                    best_name, X_train, y_train, n_trials=n_trials
                )

                if best_params:
                    tuned_models = train_model_with_params(
                        X_train, y_train, best_name, best_params
                    )

                    tuned_model = tuned_models[best_name]

                    # Evaluar si realmente mejora vs baseline
                    logger.info("Evaluando mejora del modelo optimizado...")

                    baseline_score = cv_results.loc[
                        cv_results["model"] == best_name
                    ].iloc[0]

                    # Seleccionar métrica según problema
                    if "f1_mean" in baseline_score:
                        metric_name = "f1_mean"
                        scoring = "f1_weighted"
                    else:
                        metric_name = "r2_mean"
                        scoring = "r2"

                    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

                    if metric_name == "f1_mean":
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    else:
                        cv = KFold(n_splits=3, shuffle=True, random_state=42)

                    tuned_score = cross_val_score(
                        tuned_model,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1
                    ).mean()

                    logger.info(f"Baseline {metric_name}: {baseline_score[metric_name]:.4f}")
                    logger.info(f"Tuned {metric_name}: {tuned_score:.4f}")

                    # Solo reemplazar si mejora
                    if tuned_score > baseline_score[metric_name]:
                        final_model = tuned_model
                        logger.info("Se usa modelo OPTIMIZADO (mejora detectada)")
                    else:
                        logger.info("Se mantiene modelo BASE (no hubo mejora)")

                else:
                    logger.warning("Optuna no retornó params. Usando modelo base.")

            except Exception as e:
                logger.warning(f"Optuna falló: {e}. Continuando con modelo base.")


        # ── 7. Evaluación en test ─────────────────────
        logger.info("\n[PASO 7] Evaluando en test...")

        # Métricas de modelos base (comparación dashboard)
        test_metrics = evaluate_trained_models(
            trained_models, X_test, y_test
        )

        # Evaluar también el modelo final (base o optimizado)
        y_pred_final = final_model.predict(X_test)

        if problem_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score

            final_metrics = {
                "model": f"{best_name} (final)",
                "accuracy": accuracy_score(y_test, y_pred_final),
                "f1": f1_score(y_test, y_pred_final, average="weighted", zero_division=0),
            }
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import numpy as np

            final_metrics = {
                "model": f"{best_name} (final)",
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_final)),
                "mae": mean_absolute_error(y_test, y_pred_final),
                "r2": r2_score(y_test, y_pred_final),
            }

        # Agregar a tabla
        import pandas as pd
        test_metrics = pd.concat([test_metrics, pd.DataFrame([final_metrics])], ignore_index=True)

        # Classification report del modelo final
        clf_report = get_classification_report(
            final_model, X_test, y_test, label_encoder
        )

        logger.info(f"\nClassification Report ({best_name} FINAL):\n{clf_report}")
        # ── 8. Explicabilidad ─────────────────────────
        logger.info("\n[PASO 8] Generando explicaciones (SHAP + LIME)...")

        if X_train.shape[0] > 5000:
            logger.info("Dataset grande: limitando SHAP sample")
            shap_sample_size = 50
        else:
            shap_sample_size = 100

        shap_result = compute_shap_values(
            pipeline=best_model,
            X_train=X_train,
            label_encoder=label_encoder,
            sample_size=shap_sample_size
        )
        shap_importance = get_shap_feature_importance(shap_result)

        lime_sample = X_train.sample(min(1000, len(X_train)), random_state=42)

        lime_result = compute_lime_explanation(
            pipeline=best_model,
            X_train=lime_sample,
            label_encoder=label_encoder,
        )
        lime_text = generate_lime_text_explanation(lime_result)
        logger.info(f"\n{lime_text}")

        # ── 9. PDF ────────────────────────────────────
        logger.info("\n[PASO 9] Generando reporte PDF...")
        pdf_path = os.path.join(task_output_dir, "report.pdf")

        generate_pdf_report(
            output_path=pdf_path,
            target=target,
            best_model=best_name,
            metrics_df=test_metrics,
            lime_text=lime_text,
            problem_type=problem_type,
        )
        logger.info(f"PDF: {pdf_path}")

        # ── 10. Construir resultado ───────────────────
        elapsed = (datetime.now() - start_time).total_seconds()

        results = {
            "status": "success",

            # ── Metadata ─────────────────────────────
            "run_info": {
                "file": os.path.basename(file_path),
                "target": target,
                "problem_type": problem_type,
                "best_model": best_name,
                "elapsed_seconds": round(elapsed, 2),
            },

            # ── Preprocesamiento ──────────────────────
            "preprocessing": preprocessing_report,

            # ── Rendimiento de modelos ────────────────
            "model_performance": {
                "cv_results": cv_results.to_dict(orient="records"),
                "test_metrics": test_metrics.to_dict(orient="records"),
                "classification_report": clf_report,
            },

            # ── Explicabilidad ────────────────────────
            "explainability": {
                "lime": {
                    "text": lime_text,
                    "predicted_class": lime_result.get("predicted_class"),
                    "probabilities": lime_result.get("probabilities", {}),
                },
                "shap": {
                    "feature_importance": shap_importance,
                    "explainer_type": shap_result.get("explainer_type"),
                    "status": (
                        "ok" if "error" not in shap_result
                        else shap_result["error"]
                    ),
                },
            },

            # ── Optimización ──────────────────────────
            "optimization": {
                "enabled": use_optuna,
                "best_params": best_params,
                "trials": n_trials if use_optuna else 0,
            },

            # ── Archivos de salida ────────────────────
            "output_files": {
                "pdf": pdf_path,
                "json": os.path.join(task_output_dir, "results.json"),
            },
        }

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"Pipeline falló: {e}")
        logger.error(traceback.format_exc())

        results = {
            "status": "error",
            "run_info": {
                "file": os.path.basename(file_path),
                "elapsed_seconds": round(elapsed, 2),
            },
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # Siempre guardar JSON (éxito o error)
    json_path = os.path.join(task_output_dir, "results.json")
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=str)

    os.replace(tmp_path, json_path)

    logger.info(f"\nJSON: {json_path}")
    logger.info(f"Status: {results.get('status')}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python run_analysis.py <archivo.csv> [--optuna] [--trials N]")
        sys.exit(1)

    _use_optuna = "--optuna" in sys.argv
    _n_trials = 20
    if "--trials" in sys.argv:
        idx = sys.argv.index("--trials")
        if idx + 1 < len(sys.argv):
            _n_trials = int(sys.argv[idx + 1])

    output = main(sys.argv[1], use_optuna=_use_optuna, n_trials=_n_trials)

    print(f"\nResultado: {output.get('status')}")
    if output.get("status") == "success":
        info = output.get("run_info", {})
        print(f"Mejor modelo:  {info.get('best_model')}")
        print(f"Tiempo:        {info.get('elapsed_seconds')}s")