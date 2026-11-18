"""PathSnap FastAPI application."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.trips import router

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Scores road candidates and reconstructs a route from a trip ID.",
    version="1.0.0",
)
origins = [item.strip() for item in os.environ.get(
    "PATHSNAP_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
).split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health", tags=["Operations"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pathsnap-api"}


@app.get("/", tags=["Operations"])
def root() -> dict[str, str]:
    return {"service": "PathSnap Map-Matching API", "docs": "/docs", "health": "/health"}
