"""TruthLens AI - FastAPI application.

Run:  uvicorn app.main:app --port 8000      (from the backend folder)
Docs: http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .database import init_db
from .routers import batch, image, system, text
from .services.common import ModelNotReadyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("truthlens")
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    logger.info("TruthLens AI backend started. Models load on first use.")
    yield


app = FastAPI(
    title="TruthLens AI API",
    description="Explainable multilingual multimodal fake news and AI-generated image detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(_: Request, exc: ModelNotReadyError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}: {exc}"})


for router in (system.router, text.router, image.router, batch.router):
    app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": "TruthLens AI API is running. Open /docs for the API documentation."}
