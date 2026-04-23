# dashboard/streamlit_app.py

import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="Data Analyst Agent", layout="wide")

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "task_id" not in st.session_state:
    st.session_state.task_id = None

if "last_status" not in st.session_state:
    st.session_state.last_status = None


# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────

def upload_file(file, optimize):
    try:
        res = requests.post(
            API_URL + "/",
            files={"file": (file.name, file.getvalue(), file.type)},
            params={"optimize": str(optimize).lower()},
            timeout=60,
        )
        res.raise_for_status()
        return res.json()
    except requests.exceptions.Timeout:
        st.error("Timeout al subir archivo.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión: {e}")
    return None


def get_status(task_id):
    try:
        res = requests.get(f"{API_URL}/status/{task_id}", timeout=10)
        res.raise_for_status()
        return res.json()
    except:
        return None


def get_results(task_id):
    try:
        res = requests.get(f"{API_URL}/results/{task_id}", timeout=20)
        res.raise_for_status()
        return res.json()
    except:
        return None


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("Data Analyst Agent")

# ─────────────────────────────────────────────
# REGLAS DE NEGOCIO (CRÍTICO)
# ─────────────────────────────────────────────

with st.expander("Reglas de uso y consideraciones del sistema", expanded=False):

    st.markdown("""
### Entrada de datos
- Solo se permiten archivos **CSV o Excel (.xlsx)**.
- Tamaño máximo permitido: **50 MB**.
- El dataset debe contener una columna objetivo (target).
- El sistema intentará detectar automáticamente el target, pero:
  - Si hay múltiples columnas candidatas, puede elegir incorrectamente.
  - Se recomienda usar nombres como: `target`, `label`, `class`, `outcome`.

### Procesamiento
- Cada análisis se ejecuta en un **proceso independiente**.
- El sistema soporta un máximo de:
  - **2 ejecuciones simultáneas**
  - **10 tareas en cola**
- Si el sistema está saturado, la solicitud será rechazada.

### Modelado
- Se evalúan múltiples modelos automáticamente:
  - RandomForest, LightGBM, SVM, KNN, MLP, DecisionTree
- Se selecciona el mejor modelo basado en:
  - Clasificación → F1-score
  - Regresión → R²

### Optimización (Optuna)
- Si se activa:
  - Solo se optimiza el **mejor modelo encontrado**
  - Aumenta significativamente el tiempo de ejecución
- No garantiza mejora, pero suele mejorar resultados en datasets complejos

### Explicabilidad
- Se generan explicaciones con:
  - SHAP → importancia global
  - LIME → explicación local
- Puede fallar en datasets muy grandes o con alta dimensionalidad

### Salidas
- Se genera:
  - Métricas de modelos
  - Explicaciones
  - Reporte PDF descargable

### Limitaciones
- No soporta:
  - Datos no estructurados (texto libre sin procesar)
  - Imágenes
- Performance puede degradarse con:
  - Muchas columnas categóricas
  - Alta cardinalidad
  - Datos muy grandes

### Buenas prácticas
- Limpiar datos antes de subirlos
- Evitar columnas ID o irrelevantes
- Reducir cardinalidad si es posible
""")

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader("Archivo de entrada", type=["csv", "xlsx"])
optimize = st.checkbox("Optimizar con Optuna", value=False)

if st.button("Ejecutar análisis", use_container_width=True):

    if not uploaded_file:
        st.warning("Debes subir un archivo.")
        st.stop()

    response = upload_file(uploaded_file, optimize)

    if response:
        st.session_state.task_id = response["task_id"]
        st.success(f"Task creada: {st.session_state.task_id}")


# ─────────────────────────────────────────────
# TRACKING
# ─────────────────────────────────────────────

if st.session_state.task_id:

    task_id = st.session_state.task_id

    st.subheader("Estado del análisis")

    if st.button("Actualizar estado"):
        status = get_status(task_id)
        st.session_state.last_status = status

    status = st.session_state.last_status

    if not status:
        st.info("Presiona 'Actualizar estado' para consultar progreso.")
        st.stop()

    state = status.get("status")

    if state == "queued":
        st.info("En cola")

    elif state == "running":
        st.warning("Procesando...")

    elif state == "failed":
        st.error(status.get("error"))

    elif state == "completed":
        st.success("Completado")

    # ─────────────────────────────────────────
    # RESULTADOS
    # ─────────────────────────────────────────

    if state == "completed":

        results = get_results(task_id)

        if not results:
            st.error("No se pudieron obtener resultados.")
            st.stop()

        run_info = results.get("run_info", {})
        problem_type = run_info.get("problem_type", "classification")

        st.subheader("Resumen")

        col1, col2, col3 = st.columns(3)
        col1.metric("Modelo", run_info.get("best_model"))
        col2.metric("Target", run_info.get("target"))
        col3.metric("Tiempo (s)", run_info.get("elapsed_seconds"))

        st.write(f"Tipo de problema: {problem_type}")

        # ─────────────────────────────────────
        # MÉTRICAS
        # ─────────────────────────────────────

        st.subheader("Métricas")

        perf = results.get("model_performance", {})
        df_metrics = pd.DataFrame(perf.get("test_metrics", []))

        if not df_metrics.empty:
            st.dataframe(df_metrics, use_container_width=True)

        if problem_type == "classification":
            clf_report = perf.get("classification_report")
            if clf_report:
                with st.expander("Classification Report"):
                    st.code(clf_report)

        # ─────────────────────────────────────
        # LIME
        # ─────────────────────────────────────

        st.subheader("LIME")

        lime_data = results.get("explainability", {}).get("lime", {})
        lime_text = lime_data.get("text")

        if lime_text:
            st.code(lime_text)

            if problem_type == "classification":
                probs = lime_data.get("probabilities", {})
                if probs:
                    st.bar_chart(pd.Series(probs))

        # ─────────────────────────────────────
        # SHAP
        # ─────────────────────────────────────

        st.subheader("SHAP")

        shap_data = results.get("explainability", {}).get("shap", {})
        shap_importance = shap_data.get("feature_importance")

        if shap_importance:
            df_shap = pd.DataFrame(shap_importance)
            st.bar_chart(df_shap.set_index("feature")["importance"])

        # ─────────────────────────────────────
        # PDF
        # ─────────────────────────────────────

        pdf_url = f"{API_URL}/download/{task_id}"
        st.markdown(f"[Descargar PDF]({pdf_url})")