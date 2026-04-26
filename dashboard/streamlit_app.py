# dashboard/streamlit_app.py

import time
import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")
POLL_INTERVAL = 3

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

defaults = {
    "task_id": None,
    "task_status": None,
    "results": None,
    "polling": False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────

def upload_file(file, optimize: bool):
    try:
        res = requests.post(
            API_URL + "/",
            files={"file": (file.name, file.getvalue(), file.type)},
            params={"optimize": str(optimize).lower()},
            timeout=60,
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Error subiendo archivo: {e}")
        return None


def fetch_status(task_id: str):
    try:
        res = requests.get(f"{API_URL}/status/{task_id}", timeout=10)
        res.raise_for_status()
        return res.json()
    except:
        return None


def fetch_results(task_id: str):
    try:
        res = requests.get(f"{API_URL}/results/{task_id}", timeout=20)
        res.raise_for_status()
        return res.json()
    except:
        return None


# ─────────────────────────────────────────────
# STATUS CONFIG
# ─────────────────────────────────────────────

STATUS_CONFIG = {
    "queued":    {"pct": 5,   "label": "En cola..."},
    "running":   {"pct": 60,  "label": "Ejecutando pipeline..."},
    "completed": {"pct": 100, "label": "Completado"},
    "failed":    {"pct": 100, "label": "Falló"},
}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🤖 Data Analyst Agent")
st.divider()

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

if not st.session_state.polling and st.session_state.task_status != "completed":

    uploaded_file = st.file_uploader("Sube dataset", type=["csv", "xlsx"])
    optimize = st.checkbox("Optimizar con Optuna")

    if st.button("▶ Ejecutar análisis", use_container_width=True):
        if not uploaded_file:
            st.warning("Sube un archivo")
            st.stop()

        response = upload_file(uploaded_file, optimize)

        if response:
            st.session_state.task_id = response["task_id"]
            st.session_state.task_status = "queued"
            st.session_state.polling = True
            st.session_state.results = None
            st.rerun()

# ─────────────────────────────────────────────
# POLLING
# ─────────────────────────────────────────────

if st.session_state.polling and st.session_state.task_id:

    task_id = st.session_state.task_id
    status_data = fetch_status(task_id)

    if status_data:
        st.session_state.task_status = status_data.get("status")

    state = st.session_state.task_status
    cfg = STATUS_CONFIG.get(state, {"pct": 0, "label": "..."})

    st.subheader("Progreso")
    st.progress(cfg["pct"])
    st.info(cfg["label"])

    if state == "completed":
        st.session_state.polling = False

        if not st.session_state.results:
            st.session_state.results = fetch_results(task_id)

        st.rerun()

    elif state == "failed":
        st.session_state.polling = False
        st.error(status_data.get("error", "Error desconocido"))

    else:
        time.sleep(POLL_INTERVAL)
        st.rerun()

# ─────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────

if st.session_state.task_status == "completed" and st.session_state.results:

    results = st.session_state.results
    run_info = results.get("run_info", {})

    st.subheader("Resultados")

    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo", run_info.get("best_model"))
    col2.metric("Target", run_info.get("target"))
    col3.metric("Tiempo", f"{run_info.get('elapsed_seconds')}s")

    # ── MÉTRICAS
    st.subheader("Métricas")

    test_metrics = results.get("model_performance", {}).get("test_metrics", [])
    if test_metrics:
        df = pd.DataFrame(test_metrics)
        st.dataframe(df, use_container_width=True)

    # ── SHAP
    st.subheader("SHAP")

    shap_imp = results.get("explainability", {}).get("shap", {}).get("feature_importance")

    if shap_imp:
        df = pd.DataFrame(shap_imp)
        st.bar_chart(df.set_index("feature")["importance"])

    # ── LIME
    st.subheader("LIME")

    lime = results.get("explainability", {}).get("lime", {})
    if lime.get("text"):
        st.code(lime["text"])

        probs = lime.get("probabilities")
        if probs:
            st.bar_chart(pd.Series(probs))

    # ── PREPROCESSING
    with st.expander("Preprocesamiento"):

        prep = results.get("preprocessing", {})

        st.write("Numéricas:")
        st.dataframe(pd.DataFrame(prep.get("numeric_features", []), columns=["column"]))

        st.write("Nominales:")
        st.dataframe(pd.DataFrame(prep.get("nominal_features", []), columns=["column"]))

        st.write("Ordinales:")
        st.dataframe(pd.DataFrame(prep.get("ordinal_features", []), columns=["column"]))

        st.write("Descartadas:")
        descartadas = (
            prep.get("dropped_high_missing", [])
            + prep.get("dropped_high_cardinality", [])
            + prep.get("dropped_id_like", [])
        )
        st.dataframe(pd.DataFrame(descartadas, columns=["column"]))

    # ── PDF
    st.markdown(f"[Descargar PDF]({API_URL}/download/{st.session_state.task_id})")

    # ── RESET
    if st.button("Nuevo análisis"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()