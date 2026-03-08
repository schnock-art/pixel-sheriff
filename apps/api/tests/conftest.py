import os
import tempfile

from httpx import ASGITransport, AsyncClient
import pytest_asyncio

_TEST_RUN_ID = str(os.getpid())
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:////{tempfile.gettempdir().lstrip('/')}/pixel_sheriff_test_{_TEST_RUN_ID}.db")
os.environ.setdefault("STORAGE_ROOT", f"{tempfile.gettempdir()}/pixel_sheriff_test_data_{_TEST_RUN_ID}")

from sheriff_api.db.models import Base
from sheriff_api.db.session import engine
from sheriff_api.main import app


@pytest_asyncio.fixture(autouse=True)
async def reset_db() -> None:
    async with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            await conn.exec_driver_sql("DROP SCHEMA IF EXISTS public CASCADE")
            await conn.exec_driver_sql("CREATE SCHEMA public")
        else:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as test_client:
            yield test_client
