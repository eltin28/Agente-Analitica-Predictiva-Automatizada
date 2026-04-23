# dashboard/streamlit_app.py
"""
Dashboard del agente de análisis automático.
Polling automático cada 3 segundos sin botón manual.
Muestra barra de progreso reactiva según estado de la task.
"""

import time
import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/analyze"

POLL_INTERVAL = 3   # segundos entre consultas de estado

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

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
    "task_status": None,    # queued | running | completed | failed
    "results": None,        # dict con resultados cuando completed
    "polling": False,       # True mientras el pipeline está activo
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────

def upload_file(file, optimize: bool) -> dict | None:
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
        st.error("Timeout al subir archivo. Reintenta.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión: {e}")
    return None


def fetch_status(task_id: str) -> dict | None:
    try:
        res = requests.get(f"{API_URL}/status/{task_id}", timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None


def fetch_results(task_id: str) -> dict | None:
    try:
        res = requests.get(f"{API_URL}/results/{task_id}", timeout=20)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None

# ─────────────────────────────────────────────
# ESTADO → PROGRESO
# ─────────────────────────────────────────────

STATUS_CONFIG = {
    "queued":    {"pct": 5,   "label": "En cola — esperando proceso disponible..."},
    "running":   {"pct": 55,  "label": "Ejecutando pipeline de ML..."},
    "completed": {"pct": 100, "label": "Análisis completado"},
    "failed":    {"pct": 100, "label": "El análisis falló"},
}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🤖 Data Analyst Agent")

with st.expander("Capacidades y limitaciones del sistema", expanded=False):
    st.markdown("""
**Pipeline automático CRISP-DM**
- Detección automática de variable objetivo (`target`, `label`, `class`, `survived`…)
- Preprocesamiento dinámico: imputación, encoding, winsorización, filtro de correlación
- 6 modelos: RandomForest · LightGBM · SVM · KNN · MLP · DecisionTree
- Selección automática por F1-weighted con validación cruzada (5 folds)
- Desbalanceo manejado con SMOTE
- Interpretabilidad: SHAP (global) + LIME (local)
- Optimización opcional con Optuna (solo sobre el mejor modelo)
- Reporte PDF ejecutivo descargable

**Restricciones**
- Formatos: CSV o XLSX · Máximo 50 MB
- Máx. 2 análisis simultáneos · 10 en cola
- No soporta texto libre, imágenes ni series de tiempo
- El procesamiento puede tardar varios minutos
""")

st.divider()

# ─────────────────────────────────────────────
# INPUT — solo visible si no hay tarea activa
# ─────────────────────────────────────────────

if not st.session_state.polling and st.session_state.task_status != "completed":

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Sube tu dataset (CSV o XLSX)",
            type=["csv", "xlsx"],
        )
    with col2:
        st.write("")   # espaciado vertical
        st.write("")
        optimize = st.checkbox("Optimizar con Optuna", value=False)

    if st.button("▶ Ejecutar análisis", use_container_width=True, type="primary"):
        if not uploaded_file:
            st.warning("Debes subir un archivo antes de continuar.")
            st.stop()

        with st.spinner("Enviando archivo..."):
            response = upload_file(uploaded_file, optimize)

        if response:
            st.session_state.task_id    = response["task_id"]
            st.session_state.task_status = "queued"
            st.session_state.polling    = True
            st.session_state.results    = None
            st.rerun()

# ─────────────────────────────────────────────
# POLLING AUTOMÁTICO
# ─────────────────────────────────────────────

if st.session_state.polling and st.session_state.task_id:

    task_id = st.session_state.task_id
    status_data = fetch_status(task_id)

    if status_data:
        st.session_state.task_status = status_data.get("status", "unknown")

    state = st.session_state.task_status
    cfg   = STATUS_CONFIG.get(state, {"pct": 0, "label": "Desconocido"})

    # ── Barra de progreso ─────────────────────
    st.subheader("Progreso del análisis")

    if state == "running":
        # Barra indeterminada animada mientras corre el pipeline
        progress_bar = st.progress(0)
        status_text  = st.empty()
        status_text.info(f"⚙️  {cfg['label']}")

        # Animación suave: avanza de 5% a 90% en incrementos
        for pct in range(5, 91, 2):
            time.sleep(0.04)
            progress_bar.progress(pct)

        # Consultar estado real al terminar la animación
        status_data = fetch_status(task_id)
        if status_data:
            st.session_state.task_status = status_data.get("status", state)

        st.rerun()

    elif state == "queued":
        st.progress(cfg["pct"])
        st.info(f"⏳  {cfg['label']}")
        time.sleep(POLL_INTERVAL)
        st.rerun()

    elif state == "completed":
        st.progress(100)
        st.success(f"✅  {cfg['label']}")
        st.session_state.polling = False

        # Cargar resultados una sola vez
        if not st.session_state.results:
            with st.spinner("Cargando resultados..."):
                st.session_state.results = fetch_results(task_id)
        st.rerun()

    elif state == "failed":
        st.progress(100)
        err = (status_data or {}).get("error", "Error desconocido")
        st.error(f"❌  {cfg['label']}: {err}")
        st.session_state.polling = False

        if st.button("🔄 Intentar de nuevo"):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

# ─────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────

if st.session_state.task_status == "completed" and st.session_state.results:

    results  = st.session_state.results
    run_info = results.get("run_info", {})

    st.divider()
    st.subheader("Resultados del análisis")

    # ── KPIs principales ──────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo seleccionado", run_info.get("best_model", "—"))
    col2.metric("Variable objetivo",   run_info.get("target", "—"))
    col3.metric("Tiempo de ejecución", f"{run_info.get('elapsed_seconds', '—')} s")

    # ── Optuna ───────────────────────────────
    optim = results.get("optimization", {})
    if optim.get("enabled") and optim.get("best_params"):
        with st.expander("⚙️ Parámetros encontrados por Optuna"):
            st.json(optim["best_params"])

    # ── Métricas ──────────────────────────────
    st.subheader("Comparación de modelos")
    perf        = results.get("model_performance", {})
    test_metrics = perf.get("test_metrics", [])

    if test_metrics:
        df_m = pd.DataFrame(test_metrics)
        best = run_info.get("best_model", "")

        # Resaltar fila del mejor modelo
        def highlight_best(row):
            color = "background-color: #d4edda" if row["model"] == best else ""
            return [color] * len(row)

        st.dataframe(
            df_m.style.apply(highlight_best, axis=1),
            use_container_width=True,
        )
    else:
        st.warning("No hay métricas disponibles.")

    clf_report = perf.get("classification_report")
    if clf_report:
        with st.expander("📋 Classification Report completo"):
            st.code(clf_report)

    # ── SHAP ──────────────────────────────────
    st.subheader("Importancia global de variables (SHAP)")
    shap_d = results.get("explainability", {}).get("shap", {})
    shap_imp = shap_d.get("feature_importance")

    if shap_imp:
        df_shap = pd.DataFrame(shap_imp)
        st.bar_chart(
            df_shap.set_index("feature")["importance"].sort_values(),
            use_container_width=True,
        )
        with st.expander("Ver tabla de importancias"):
            st.dataframe(df_shap, use_container_width=True)
    else:
        st.warning("SHAP no disponible.")

    # ── LIME ──────────────────────────────────
    st.subheader("Explicación local (LIME)")
    lime_d    = results.get("explainability", {}).get("lime", {})
    lime_text = lime_d.get("text")

    if lime_text:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.code(lime_text)
        with col2:
            probs = lime_d.get("probabilities", {})
            if probs:
                st.caption("Probabilidades por clase")
                st.bar_chart(pd.Series(probs), use_container_width=True)
    else:
        st.warning("LIME no disponible.")

    # ── Preprocesamiento ──────────────────────
    with st.expander("🔧 Detalle de preprocesamiento"):
        prep = results.get("preprocessing", {})
        if prep:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Columnas utilizadas**")
                st.write("Numéricas:",  prep.get("numeric_features", []))
                st.write("Nominales:", prep.get("nominal_features", []))
                st.write("Ordinales:", prep.get("ordinal_features", []))
            with c2:
                st.markdown("**Columnas descartadas**")
                st.write("Alta tasa de nulos:",    prep.get("dropped_high_missing", []))
                st.write("Alta cardinalidad:",     prep.get("dropped_high_cardinality", []))
                st.write("Posibles IDs:",          prep.get("dropped_id_like", []))

    # ── PDF ───────────────────────────────────
    st.divider()
    pdf_url = f"{API_URL}/download/{st.session_state.task_id}"
    st.markdown(f"### 📄 [Descargar reporte PDF]({pdf_url})")

    # ── Nuevo análisis ────────────────────────
    st.divider()
    if st.button("🔄 Analizar otro dataset"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()