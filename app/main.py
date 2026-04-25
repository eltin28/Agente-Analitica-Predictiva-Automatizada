# app/main.py
"""
FastAPI app con gestión correcta del ciclo de vida del ProcessPoolExecutor.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analyze import router as analyze_router, get_executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida:
      - startup:  inicializa el ProcessPoolExecutor
      - shutdown: espera que terminen los procesos activos y cierra limpio
    """
    logger.info("Iniciando ProcessPoolExecutor...")
    executor = get_executor()   # inicializa el pool
    logger.info("API lista.")

    yield  # app corriendo

    logger.info("Cerrando ProcessPoolExecutor (esperando procesos activos)...")
    executor.shutdown(wait=True)
    logger.info("Shutdown completo.")


app = FastAPI(
    title="Data Analyst Agent API",
    description="Pipeline CRISP-DM automático: sube un CSV, recibe modelo + reporte.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit local
        "http://127.0.0.1:8501",
        # agrega aquí tu dominio real en producción
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/analyze", tags=["análisis"])


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", reload=True)