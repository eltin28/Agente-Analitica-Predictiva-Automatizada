# app/routes/analyze.py
"""
Endpoints de análisis con ejecución asíncrona real.

Reglas de negocio:
  1. Solo CSV y Excel (.xlsx), máximo MAX_FILE_MB
  2. Cada análisis en proceso aislado (ProcessPoolExecutor)
  3. Máximo MAX_WORKERS análisis simultáneos
  4. Cola limitada a MAX_QUEUE_SIZE tareas activas
  5. Ciclo de vida: queued → running → completed | failed
  6. Estado persistido en disco (JSON atómico)
  7. Archivo eliminado automáticamente al terminar
  8. PDF accesible por task_id vía GET /download/{task_id}

Flujo:
  POST /analyze           → retorna task_id inmediatamente
  GET  /status/{task_id}  → estado actual
  GET  /results/{task_id} → resultado final
  GET  /download/{task_id}→ descarga PDF
"""

import os
import json
import uuid
import logging
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

UPLOAD_DIR       = "outputs/uploads"
TASK_DIR         = "outputs/tasks"
TASK_OUTPUT_DIR  = "outputs/tasks_output"   # PDFs y JSONs por task

MAX_WORKERS    = 2
MAX_QUEUE_SIZE = 10
MAX_FILE_MB    = 50
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

_executor: ProcessPoolExecutor | None = None


def get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    return _executor


# ─────────────────────────────────────────────
# GESTIÓN DE TASKS (JSON store)
# ─────────────────────────────────────────────

def _task_path(task_id: str) -> str:
    return os.path.join(TASK_DIR, f"{task_id}.json")


def _write_task(task_id: str, data: dict) -> None:
    """Escritura atómica: .tmp → rename para nunca tener JSON corrupto."""
    os.makedirs(TASK_DIR, exist_ok=True)
    tmp = _task_path(task_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, _task_path(task_id))


def _read_task(task_id: str) -> dict | None:
    path = _task_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _count_active_tasks() -> int:
    if not os.path.exists(TASK_DIR):
        return 0

    count = 0
    for fname in os.listdir(TASK_DIR):
        if not fname.endswith(".json"):
            continue

        try:
            path = os.path.join(TASK_DIR, fname)
            with open(path, encoding="utf-8") as f:
                status = json.load(f).get("status")

            if status in ("queued", "running"):
                count += 1

        except Exception:
            continue

    return count


# ─────────────────────────────────────────────
# FUNCIÓN QUE CORRE EN EL PROCESO SEPARADO
# ─────────────────────────────────────────────

def _run_pipeline_in_process(
    task_id: str,
    file_path: str,
    optimize: bool
) -> None:
    """
    Ejecutada en proceso hijo. Importa el pipeline aquí para evitar
    problemas de serialización con fork en Windows.
    """
    from run_analysis import main as run_pipeline

    task_data = _read_task(task_id) or {}

    try:
        task_data.update({
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
        })
        _write_task(task_id, task_data)

        result = run_pipeline(file_path, use_optuna=optimize)

        final_status = "completed" if result.get("status") == "success" else "failed"
        task_data.update({
            "status": final_status,
            "finished_at": datetime.utcnow().isoformat(),
            "result": result,
        })
        _write_task(task_id, task_data)

    except Exception as e:
        import traceback
        task_data.update({
            "status": "failed",
            "finished_at": datetime.utcnow().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        _write_task(task_id, task_data)

    finally:
        # Limpiar archivo subido
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


# ─────────────────────────────────────────────
# POST /analyze
# ─────────────────────────────────────────────

@router.post("")
async def analyze(
    file: UploadFile = File(...),
    optimize: bool = Query(False, description="Activar optimización con Optuna"),
):
    """
    Recibe un archivo y lanza el pipeline en background.
    Retorna task_id inmediatamente (< 200ms).
    """
    filename = file.filename or ""

    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Formato no soportado. Usa CSV o Excel (.xlsx)."
        )

    # ── Verificar cola ────────────────────────────────
    if _count_active_tasks() >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=429,
            detail="Servidor ocupado. Intenta en unos minutos."
        )

    # ── Guardar archivo por chunks ────────────────────
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    task_id = str(uuid.uuid4())
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{safe_name}")

    size = 0
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB por chunk
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                f.close()
                os.remove(file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Archivo demasiado grande. Máximo: {MAX_FILE_MB} MB."
                )
            f.write(chunk)

    size_mb = round(size / (1024 * 1024), 2)

    # ── Crear task ────────────────────────────────────
    task_data = {
        "task_id": task_id,
        "status": "queued",
        "filename": filename,
        "file_size_mb": size_mb,
        "optimize": optimize,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    _write_task(task_id, task_data)

    executor = get_executor()

    if _count_active_tasks() >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=429,
            detail="Servidor ocupado. Intenta en unos minutos."
        )
    # ── Lanzar proceso ────────────────────────────────
    executor.submit(_run_pipeline_in_process, task_id, file_path, optimize)
    logger.info(f"Workers activos: {_count_active_tasks()}/{MAX_QUEUE_SIZE}")

    return {
        "task_id": task_id,
        "status": "queued",
        "status_url": f"/analyze/status/{task_id}",
        "results_url": f"/analyze/results/{task_id}",
        "download_url": f"/analyze/download/{task_id}",
    }


# ─────────────────────────────────────────────
# GET /status/{task_id}
# ─────────────────────────────────────────────

@router.get("/status/{task_id}")
def get_status(task_id: str):
    task = _read_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task no encontrada.")

    response = {
        "task_id": task_id,
        "status": task["status"],
        "filename": task.get("filename"),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
    }

    if task["status"] == "failed":
        response["error"] = task.get("error", "Error desconocido")

    if task["status"] == "completed" and task.get("result"):
        run_info = task["result"].get("run_info", {})
        response["best_model"] = run_info.get("best_model")
        response["elapsed_seconds"] = run_info.get("elapsed_seconds")

    return response


# ─────────────────────────────────────────────
# GET /results/{task_id}
# ─────────────────────────────────────────────

@router.get("/results/{task_id}")
def get_results(task_id: str):
    task = _read_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task no encontrada.")

    status = task["status"]

    if status in ("queued", "running"):
        return {"status": status, "message": "El análisis está en progreso."}

    if status == "failed":
        return {
            "status": "failed",
            "error": task.get("error", "Error desconocido"),
        }

    return task.get("result", {})


# ─────────────────────────────────────────────
# GET /download/{task_id}
# ─────────────────────────────────────────────

@router.get("/download/{task_id}")
def download_pdf(task_id: str):
    """
    Descarga el PDF del reporte ejecutivo de la task.
    Solo disponible cuando status = completed.
    """
    task = _read_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task no encontrada.")

    if task["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"El análisis no ha terminado. Estado actual: {task['status']}"
        )

    # Ruta del PDF generado por run_analysis
    pdf_path = os.path.join(TASK_OUTPUT_DIR, task_id, "report.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="PDF no encontrado. Es posible que haya sido eliminado."
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"report_{task_id[:8]}.pdf",
    )


# ─────────────────────────────────────────────
# GET /tasks
# ─────────────────────────────────────────────

@router.get("/tasks")
def list_tasks(limit: int = 20):
    """Lista las últimas N tasks. Útil para el historial del dashboard."""
    if not os.path.exists(TASK_DIR):
        return {"tasks": []}

    tasks = []
    for fname in os.listdir(TASK_DIR):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        try:
            with open(os.path.join(TASK_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
            tasks.append({
                "task_id": data.get("task_id"),
                "status": data.get("status"),
                "filename": data.get("filename"),
                "created_at": data.get("created_at"),
                "best_model": (
                    data["result"].get("run_info", {}).get("best_model")
                    if data.get("result") else None
                ),
                "elapsed_seconds": (
                    data["result"].get("run_info", {}).get("elapsed_seconds")
                    if data.get("result") else None
                ),
            })
        except Exception:
            continue

    tasks.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"tasks": tasks[:limit]}