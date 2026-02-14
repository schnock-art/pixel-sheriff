from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sheriff_api.config import get_settings
from sheriff_api.db.models import Base
from sheriff_api.db.session import engine
from sheriff_api.routers import annotations, assets, categories, exports, health, models, projects

settings = get_settings()


def parse_cors_origins(raw: str) -> list[str]:
    origins = [origin.strip() for origin in raw.split(",")]
    return [origin for origin in origins if origin]


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="pixel-sheriff", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(settings.cors_origins),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
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
