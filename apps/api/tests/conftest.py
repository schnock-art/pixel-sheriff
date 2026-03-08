import os
import sqlite3
import tempfile
import types

from httpx import ASGITransport, AsyncClient
import pytest_asyncio

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
