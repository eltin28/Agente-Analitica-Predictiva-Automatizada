# dashboard/streamlit_app.py

import time
import threading
import streamlit as st
import requests
import pandas as pd
import os
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000/analyze").rstrip("/")

# Timeouts separados por tipo de operación
UPLOAD_TIMEOUT  = 15   # solo debe recibir el archivo y retornar task_id
STATUS_TIMEOUT  = 10
RESULTS_TIMEOUT = 20

POLL_INTERVAL     = 3    # segundos entre polls
MAX_POLL_RETRIES  = 3    # reintentos por fallo de red antes de abortar
KEEPALIVE_SECS    = 600  # ping al backend cada 10 min

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────
# KEEP-ALIVE (evita spin-down en Render free tier)
# ─────────────────────────────────────────────

def _keepalive_loop(base_url: str, interval: int):
    """Hilo daemon que hace ping al backend para evitar spin-down."""
    while True:
        time.sleep(interval)
        try:
            requests.get(base_url + "/", timeout=8)
        except Exception:
            pass  # silencioso — es best-effort

def _start_keepalive():
    """Arranca el hilo de keep-alive una sola vez por sesión."""
    if "keepalive_started" not in st.session_state:
        t = threading.Thread(
            target=_keepalive_loop,
            args=(API_BASE, KEEPALIVE_SECS),
            daemon=True,
        )
        t.start()
        st.session_state.keepalive_started = True

_start_keepalive()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

DEFAULTS: dict = {
    "task_id":     None,
    "task_status": None,
    "results":     None,
    "polling":     False,
    "poll_errors": 0,    # contador de fallos consecutivos de red
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────

def _safe_post(url: str, **kwargs) -> Optional[requests.Response]:
    """POST con manejo de error unificado. Retorna Response o None."""
    try:
        res = requests.post(url, **kwargs)
        res.raise_for_status()
        return res
    except requests.exceptions.Timeout:
        st.error(
            "⏱ El servidor tardó demasiado en responder. "
            "Puede estar iniciando — espera unos segundos y reintenta."
        )
    except requests.exceptions.ConnectionError:
        st.error("🔌 No se pudo conectar al servidor. Verifica que el backend esté activo.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Error del servidor ({e.response.status_code}): {e.response.text[:200]}")
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
    return None


def _safe_get(url: str, timeout: int) -> Optional[dict]:
    """GET silencioso — retorna dict o None sin mostrar error (para polling)."""
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None


def upload_file(file, optimize: bool) -> Optional[dict]:
    """
    Sube el archivo al backend. El backend DEBE responder de inmediato
    con {"task_id": "..."} y procesar en background.
    Timeout corto (UPLOAD_TIMEOUT) porque solo transfiere el archivo.
    """
    res = _safe_post(
        API_BASE,
        files={"file": (file.name, file.getvalue(), file.type)},
        params={"optimize": str(optimize).lower()},
        timeout=UPLOAD_TIMEOUT,
    )
    return res.json() if res else None


def fetch_status(task_id: str) -> Optional[dict]:
    return _safe_get(f"{API_BASE}/status/{task_id}", timeout=STATUS_TIMEOUT)


def fetch_results(task_id: str) -> Optional[dict]:
    return _safe_get(f"{API_BASE}/results/{task_id}", timeout=RESULTS_TIMEOUT)

# ─────────────────────────────────────────────
# STATUS CONFIG
# ─────────────────────────────────────────────

STATUS_CONFIG = {
    "queued":    {"pct": 5,   "label": "En cola..."},
    "running":   {"pct": 60,  "label": "Ejecutando pipeline..."},
    "completed": {"pct": 100, "label": "Completado ✅"},
    "failed":    {"pct": 100, "label": "Falló ❌"},
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
            st.warning("Sube un archivo primero.")
            st.stop()

        with st.spinner("Enviando archivo al servidor..."):
            response = upload_file(uploaded_file, optimize)

        if response and "task_id" in response:
            st.session_state.task_id     = response["task_id"]
            st.session_state.task_status = "queued"
            st.session_state.polling     = True
            st.session_state.results     = None
            st.session_state.poll_errors = 0
            st.rerun()
        elif response:
            st.error(f"Respuesta inesperada del servidor: {response}")

# ─────────────────────────────────────────────
# POLLING
# ─────────────────────────────────────────────

if st.session_state.polling and st.session_state.task_id:

    task_id     = st.session_state.task_id
    status_data = fetch_status(task_id)

    if status_data:
        st.session_state.task_status = status_data.get("status")
        st.session_state.poll_errors = 0  # reset en éxito
    else:
        # Fallo de red: acumula reintentos antes de abortar
        st.session_state.poll_errors += 1
        if st.session_state.poll_errors >= MAX_POLL_RETRIES:
            st.session_state.polling = False
            st.error(
                f"⚠️ El servidor no respondió después de {MAX_POLL_RETRIES} intentos. "
                "El análisis puede estar corriendo en background — "
                "recarga la página en unos minutos."
            )
            st.stop()
        else:
            st.warning(
                f"⚠️ Fallo de conexión ({st.session_state.poll_errors}/{MAX_POLL_RETRIES}). "
                "Reintentando..."
            )

    state = st.session_state.task_status
    cfg   = STATUS_CONFIG.get(state, {"pct": 0, "label": "..."})

    st.subheader("Progreso")
    st.progress(cfg["pct"] / 100)
    st.info(cfg["label"])

    if state == "completed":
        st.session_state.polling = False

        if not st.session_state.results:
            st.session_state.results = fetch_results(task_id)

        st.rerun()

    elif state == "failed":
        st.session_state.polling = False
        error_msg = (status_data or {}).get("error", "Error desconocido")
        st.error(f"Pipeline falló: {error_msg}")

    else:
        time.sleep(POLL_INTERVAL)
        st.rerun()

# ─────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────

if st.session_state.task_status == "completed" and st.session_state.results:

    results  = st.session_state.results
    run_info = results.get("run_info", {})

    st.subheader("Resultados")

    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo",  run_info.get("best_model", "—"))
    col2.metric("Target",  run_info.get("target", "—"))
    col3.metric("Tiempo",  f"{run_info.get('elapsed_seconds', '—')}s")

    # ── MÉTRICAS
    st.subheader("Métricas")
    test_metrics = results.get("model_performance", {}).get("test_metrics", [])
    if test_metrics:
        df = pd.DataFrame(test_metrics)
        st.dataframe(df, use_container_width=True)
    else:
        st.caption("Sin métricas disponibles.")

    # ── SHAP
    st.subheader("SHAP")
    shap_imp = (
        results.get("explainability", {})
               .get("shap", {})
               .get("feature_importance")
    )
    if shap_imp:
        df = pd.DataFrame(shap_imp)
        st.bar_chart(df.set_index("feature")["importance"])
    else:
        st.caption("Sin datos SHAP.")

    # ── LIME
    st.subheader("LIME")
    lime = results.get("explainability", {}).get("lime", {})
    if lime.get("text"):
        st.code(lime["text"])
        probs = lime.get("probabilities")
        if probs:
            st.bar_chart(pd.Series(probs))
    else:
        st.caption("Sin datos LIME.")

    # ── PREPROCESSING
    with st.expander("Preprocesamiento"):
        prep = results.get("preprocessing", {})

        st.write("Numéricas:")
        st.dataframe(pd.DataFrame(prep.get("numeric_features", []), columns=["column"]))

        st.write("Nominales:")
        st.dataframe(pd.DataFrame(prep.get("nominal_features", []), columns=["column"]))

        st.write("Ordinales:")
        st.dataframe(pd.DataFrame(prep.get("ordinal_features", []), columns=["column"]))

        descartadas = (
            prep.get("dropped_high_missing", [])
            + prep.get("dropped_high_cardinality", [])
            + prep.get("dropped_id_like", [])
        )
        st.write("Descartadas:")
        st.dataframe(pd.DataFrame(descartadas, columns=["column"]))

    # ── PDF
    pdf_url = f"{API_BASE}/download/{st.session_state.task_id}"
    st.markdown(f"[📄 Descargar PDF]({pdf_url})")

    # ── RESET
    if st.button("🔄 Nuevo análisis"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()