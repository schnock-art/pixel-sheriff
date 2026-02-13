from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sheriff_api.config import get_settings
from sheriff_api.db.models import Base
from sheriff_api.db.session import engine
from sheriff_api.routers import annotations, assets, categories, exports, health, models, projects

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="pixel-sheriff", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")
app.include_router(categories.router, prefix="/api/v1")
app.include_router(assets.router, prefix="/api/v1")
app.include_router(annotations.router, prefix="/api/v1")
app.include_router(exports.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
