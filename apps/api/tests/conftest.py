import os
from pathlib import Path
import sqlite3
import tempfile
import types

from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio
from starlette.responses import Response

_TEST_RUN_ID = str(os.getpid())
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:////{tempfile.gettempdir().lstrip('/')}/pixel_sheriff_test_{_TEST_RUN_ID}.db")
os.environ.setdefault("STORAGE_ROOT", f"{tempfile.gettempdir()}/pixel_sheriff_test_data_{_TEST_RUN_ID}")


def _install_sqlite_test_driver_shim() -> None:
    import aiosqlite

    class _ImmediateQueue:
        def put_nowait(self, item: tuple[object, object]) -> None:
            future, function = item
            try:
                result = function()
            except BaseException as exc:
                if not future.done():
                    future.set_exception(exc)
            else:
                if not future.done():
                    future.set_result(result)

    class _FakeCursor:
        def __init__(self, cursor: sqlite3.Cursor) -> None:
            self._cursor = cursor

        @property
        def description(self):
            return self._cursor.description

        @property
        def rowcount(self) -> int:
            return self._cursor.rowcount

        @property
        def lastrowid(self) -> int:
            return self._cursor.lastrowid

        async def execute(self, *args, **kwargs):
            self._cursor.execute(*args, **kwargs)
            return self

        async def executemany(self, *args, **kwargs):
            self._cursor.executemany(*args, **kwargs)
            return self

        async def fetchall(self):
            return self._cursor.fetchall()

        async def fetchone(self):
            return self._cursor.fetchone()

        async def fetchmany(self, size: int | None = None):
            if size is None:
                return self._cursor.fetchmany()
            return self._cursor.fetchmany(size)

        async def close(self) -> None:
            self._cursor.close()

    class _FakeConnection:
        def __init__(self, database: str, **kwargs) -> None:
            kwargs.setdefault("check_same_thread", False)
            self._conn = sqlite3.connect(database, **kwargs)
            self._tx = _ImmediateQueue()
            self._thread = types.SimpleNamespace(daemon=False)

        def __await__(self):
            async def _ready():
                return self

            return _ready().__await__()

        @property
        def isolation_level(self):
            return self._conn.isolation_level

        @isolation_level.setter
        def isolation_level(self, value) -> None:
            self._conn.isolation_level = value

        async def create_function(self, *args, **kwargs) -> None:
            self._conn.create_function(*args, **kwargs)

        async def cursor(self):
            return _FakeCursor(self._conn.cursor())

        async def execute(self, *args, **kwargs):
            return _FakeCursor(self._conn.execute(*args, **kwargs))

        async def rollback(self) -> None:
            self._conn.rollback()

        async def commit(self) -> None:
            self._conn.commit()

        async def close(self) -> None:
            self._conn.close()

    def _connect(database, *args, **kwargs):
        return _FakeConnection(database, **kwargs)

    aiosqlite.connect = _connect


if os.environ["DATABASE_URL"].startswith("sqlite+aiosqlite:"):
    _install_sqlite_test_driver_shim()

from sheriff_api.db.models import Base
from sheriff_api.db.session import engine
from sheriff_api.main import app
import sheriff_api.routers.assets as assets_router
import sheriff_api.routers.datasets as datasets_router
import sheriff_api.routers.experiments.onnx as experiments_onnx_router
import sheriff_api.routers.models as models_router


class _EagerFileResponse(Response):
    def __init__(
        self,
        path: str,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        filename: str | None = None,
        background=None,
        content_disposition_type: str = "attachment",
        **_kwargs,
    ) -> None:
        response_headers = dict(headers or {})
        if filename is not None:
            response_headers.setdefault(
                "content-disposition",
                f'{content_disposition_type}; filename="{filename}"',
            )
        super().__init__(
            content=Path(path).read_bytes(),
            status_code=status_code,
            headers=response_headers,
            media_type=media_type,
            background=background,
        )


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


@pytest.fixture(autouse=True)
def eager_file_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid the sandbox-specific FileResponse thread handoff hang while preserving bytes and headers.
    for module in (assets_router, datasets_router, models_router, experiments_onnx_router):
        monkeypatch.setattr(module, "FileResponse", _EagerFileResponse)


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as test_client:
            yield test_client
