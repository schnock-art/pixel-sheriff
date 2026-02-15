from httpx import ASGITransport, AsyncClient
import pytest_asyncio

from sheriff_api.db.models import Base
from sheriff_api.db.session import engine
from sheriff_api.main import app


@pytest_asyncio.fixture(autouse=True)
async def reset_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as test_client:
            yield test_client
