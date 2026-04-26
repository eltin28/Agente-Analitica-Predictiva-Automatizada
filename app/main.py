# app/main.py

import logging
import os
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

# ── Orígenes permitidos ───────────────────────────────────────────────────────
# Se leen de variable de entorno para no hardcodear URLs de producción.
# En render.yaml se define ALLOWED_ORIGINS con el dominio real del dashboard.
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando executor...")
    executor = get_executor()
    logger.info("API lista.")
    yield
    logger.info("Cerrando executor (esperando tareas activas)...")
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
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/analyze", tags=["análisis"])


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", reload=True)