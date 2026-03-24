from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, offline, realtime


app = FastAPI(title="NoiseShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(offline.router, prefix="/api/v1", tags=["offline"])
app.include_router(realtime.router, prefix="/api/v1", tags=["realtime"])
