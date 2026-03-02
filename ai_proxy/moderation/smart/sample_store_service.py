"""
Async SampleStorage service that prevents RocksDB LOCK contention.

Design goals:
- In a *single process*, keep exactly one RocksDB handle per moderation profile.
- Serialize ALL reads/writes through a per-profile queue handled by a single execution context.
- In *multi-worker* (multi-process) deployments, elect exactly one "writer" worker process to open
  RocksDB in read-write mode; other workers proxy all reads/writes to the writer via local IPC.

This module intentionally keeps the existing sync `SampleStorage` (used by offline tools / training
subprocess) unchanged; online request paths should use the async proxy APIs here.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import queue
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from ai_proxy.moderation.smart.storage import Sample, SampleStorage


class RocksServiceUnavailable(RuntimeError):
    pass


class _InterProcessFileLock:
    """
    Cross-platform best-effort inter-process exclusive lock based on locking a file byte.
    The lock is held as long as the underlying file handle stays open.
    """

    def __init__(self, lock_path: str):
        self.lock_path = str(lock_path)
        self._fh: Optional[Any] = None

    def try_acquire(self) -> bool:
        Path(self.lock_path).parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                # Lock first byte, non-blocking.
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                fh.close()
            except Exception:
                pass
            return False

        self._fh = fh
        return True

    def release(self) -> None:
        fh = self._fh
        self._fh = None
        if fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass


@dataclass(frozen=True)
class _WorkItem:
    method: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    fut: "asyncio.Future[Any]"


class _ProfileWorker:
    """
    One profile -> one worker thread -> one SampleStorage instance -> one RocksDB handle.
    All operations are executed sequentially on that single thread.
    """

    def __init__(self, profile_name: str, db_path: str):
        self.profile_name = profile_name
        self.db_path = db_path
        self._q: "queue.Queue[object]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._thread_stop = threading.Event()
        self._storage: Optional[SampleStorage] = None
        self._open_error: Optional[BaseException] = None

    def _open_storage(self) -> None:
        if self._storage is not None or self._open_error is not None:
            return
        try:
            self._storage = SampleStorage(self.db_path, read_only=False)
        except Exception as e:
            # Fallback to read-only to keep online reads alive when an external process holds the lock.
            try:
                self._storage = SampleStorage(self.db_path, read_only=True)
            except Exception:
                self._open_error = e

    async def start(self) -> None:
        if self._thread is not None:
            return

        # Dedicated single worker thread to avoid blocking FastAPI's event loop and to
        # ensure all RocksDB operations happen in one execution context.
        def _run() -> None:
            try:
                self._thread_main()
            finally:
                if self._storage is not None:
                    # Ensure RocksDB handle is closed before process shutdown.
                    try:
                        self._storage.db.close()
                    except Exception:
                        pass

        t = threading.Thread(target=_run, name=f"RocksWorker[{self.profile_name}]", daemon=True)
        self._thread = t
        t.start()

    async def stop(self) -> None:
        self._thread_stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        t = self._thread
        if t is not None:
            t.join(timeout=2)

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[Any]" = loop.create_future()
        self._q.put(_WorkItem(method=method, args=args, kwargs=kwargs, fut=fut))
        return await fut

    def _thread_main(self) -> None:
        while not self._thread_stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            assert isinstance(item, _WorkItem)
            try:
                self._open_storage()
                if self._storage is None:
                    raise RocksServiceUnavailable(
                        f"SampleStorage open failed for profile={self.profile_name}: {self._open_error!r}"
                    )
                fn = getattr(self._storage, item.method)
                res = fn(*item.args, **item.kwargs)
                item.fut.get_loop().call_soon_threadsafe(item.fut.set_result, res)
            except Exception as e:
                item.fut.get_loop().call_soon_threadsafe(item.fut.set_exception, e)


class _IpcCodec:
    @staticmethod
    async def read_msg(reader: asyncio.StreamReader) -> dict:
        header = await reader.readexactly(4)
        (n,) = struct.unpack("!I", header)
        payload = await reader.readexactly(n)
        return pickle.loads(payload)

    @staticmethod
    async def write_msg(writer: asyncio.StreamWriter, msg: dict) -> None:
        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        writer.write(struct.pack("!I", len(payload)))
        writer.write(payload)
        await writer.drain()


class _RocksIpcServer:
    def __init__(self, service: "SampleStoreService", host: str, port: int):
        self._service = service
        self._host = host
        self._port = int(port)
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(self._handle, host=self._host, port=self._port)
        addrs = ", ".join(str(sock.getsockname()) for sock in (self._server.sockets or []))
        print(f"[ROCKS_IPC] writer server listening on {addrs}")

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                req = await _IpcCodec.read_msg(reader)
                profile_name = req["profile_name"]
                db_path = req["db_path"]
                method = req["method"]
                args = tuple(req.get("args", ()))
                kwargs = dict(req.get("kwargs", {}))

                try:
                    result = await self._service._call_local(profile_name, db_path, method, *args, **kwargs)
                    if isinstance(result, Sample):
                        result = {"__type__": "Sample", "data": result.model_dump()}
                    resp = {"ok": True, "result": result}
                except Exception as e:
                    resp = {"ok": False, "exc_type": type(e).__name__, "error": str(e)}
                await _IpcCodec.write_msg(writer, resp)
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            return
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


class _RocksIpcClient:
    def __init__(self, host: str, port: int, timeout_s: float = 2.0):
        self._host = host
        self._port = int(port)
        self._timeout_s = float(timeout_s)

    async def call(self, profile_name: str, db_path: str, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host=self._host, port=self._port),
                timeout=self._timeout_s,
            )
        except Exception as e:
            raise RocksServiceUnavailable(f"connect to rocks writer failed: {e}") from e

        try:
            await _IpcCodec.write_msg(
                writer,
                {
                    "profile_name": profile_name,
                    "db_path": db_path,
                    "method": method,
                    "args": args,
                    "kwargs": kwargs,
                },
            )
            resp = await asyncio.wait_for(_IpcCodec.read_msg(reader), timeout=self._timeout_s)
            if resp.get("ok"):
                result = resp.get("result")
                if isinstance(result, dict) and result.get("__type__") == "Sample":
                    return Sample(**(result.get("data") or {}))
                return result
            raise RocksServiceUnavailable(f"{resp.get('exc_type')}: {resp.get('error')}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


class SampleStoreService:
    """
    Per-process service facade.

    - Writer process: owns per-profile workers and RocksDB handles.
    - Non-writer process: proxies to writer via IPC; never opens RocksDB in read-write.
    """

    def __init__(self) -> None:
        self._started = False
        self._is_writer = False
        self._lock: Optional[_InterProcessFileLock] = None
        self._ipc_server: Optional[_RocksIpcServer] = None
        self._ipc_client: Optional[_RocksIpcClient] = None
        self._workers: Dict[str, _ProfileWorker] = {}
        self._workers_lock = asyncio.Lock()

    @property
    def is_writer(self) -> bool:
        return bool(self._is_writer)

    async def start(self) -> None:
        if self._started:
            return

        host = os.environ.get("AI_PROXY_ROCKS_IPC_HOST", "127.0.0.1")
        port = int(os.environ.get("AI_PROXY_ROCKS_IPC_PORT", "51234"))
        lock_path = os.environ.get("AI_PROXY_ROCKS_WRITER_LOCK", "configs/mod_profiles/.rocks_writer.lock")

        self._lock = _InterProcessFileLock(lock_path)
        self._is_writer = self._lock.try_acquire()

        if self._is_writer:
            self._ipc_server = _RocksIpcServer(self, host=host, port=port)
            await self._ipc_server.start()
            print("[ROCKS_SERVICE] role=writer (opens RocksDB read-write)")
        else:
            self._ipc_client = _RocksIpcClient(host=host, port=port)
            print("[ROCKS_SERVICE] role=client (proxy to writer; no local RocksDB opens)")

        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        if self._ipc_server is not None:
            await self._ipc_server.stop()
            self._ipc_server = None
        async with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            try:
                await w.stop()
            except Exception:
                pass
        if self._lock is not None:
            self._lock.release()
            self._lock = None
        self._started = False

    async def _get_worker(self, profile_name: str, db_path: str) -> _ProfileWorker:
        async with self._workers_lock:
            w = self._workers.get(profile_name)
            if w is None or w.db_path != db_path:
                w = _ProfileWorker(profile_name=profile_name, db_path=db_path)
                await w.start()
                self._workers[profile_name] = w
            return w

    async def _call_local(self, profile_name: str, db_path: str, method: str, *args: Any, **kwargs: Any) -> Any:
        w = await self._get_worker(profile_name, db_path)
        return await w.call(method, *args, **kwargs)

    async def call(self, profile_name: str, db_path: str, method: str, *args: Any, **kwargs: Any) -> Any:
        if not self._started:
            await self.start()

        if self._is_writer:
            return await self._call_local(profile_name, db_path, method, *args, **kwargs)
        if self._ipc_client is None:
            raise RocksServiceUnavailable("rocks ipc client not initialized")
        return await self._ipc_client.call(profile_name, db_path, method, *args, **kwargs)

    # ---- Public async APIs (keep SampleStorage semantics) ----
    async def get_sample_count(self, profile_name: str, db_path: str) -> int:
        try:
            return int(await self.call(profile_name, db_path, "get_sample_count"))
        except Exception as e:
            print(f"[ROCKS_SERVICE] get_sample_count failed: {profile_name} - {e}")
            return 0

    async def find_by_text(self, profile_name: str, db_path: str, text: str) -> Optional[Sample]:
        try:
            return await self.call(profile_name, db_path, "find_by_text", text)
        except Exception as e:
            print(f"[ROCKS_SERVICE] find_by_text failed: {profile_name} - {e}")
            return None

    async def save_sample(self, profile_name: str, db_path: str, text: str, label: int, category: Optional[str] = None) -> None:
        try:
            await self.call(profile_name, db_path, "save_sample", text, int(label), category)
        except Exception as e:
            # Never fail online requests because of RocksDB open/lock issues.
            print(f"[ROCKS_SERVICE] save_sample failed (ignored): {profile_name} - {e}")

    async def cleanup_excess_samples(self, profile_name: str, db_path: str, max_items: int) -> None:
        try:
            await self.call(profile_name, db_path, "cleanup_excess_samples", int(max_items))
        except Exception as e:
            print(f"[ROCKS_SERVICE] cleanup_excess_samples failed (ignored): {profile_name} - {e}")


class AsyncSampleStorage:
    """
    Async facade with the same method names used in the project.
    """

    def __init__(self, service: SampleStoreService, profile_name: str, db_path: str):
        self._svc = service
        self._profile_name = profile_name
        self._db_path = db_path

    async def get_sample_count(self) -> int:
        return await self._svc.get_sample_count(self._profile_name, self._db_path)

    async def find_by_text(self, text: str) -> Optional[Sample]:
        return await self._svc.find_by_text(self._profile_name, self._db_path, text)

    async def save_sample(self, text: str, label: int, category: Optional[str] = None) -> None:
        await self._svc.save_sample(self._profile_name, self._db_path, text, label, category)

    async def cleanup_excess_samples(self, max_items: int) -> None:
        await self._svc.cleanup_excess_samples(self._profile_name, self._db_path, max_items)


_service_singleton: Optional[SampleStoreService] = None


def get_sample_store_service() -> SampleStoreService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = SampleStoreService()
    return _service_singleton


async def init_sample_store_service() -> SampleStoreService:
    svc = get_sample_store_service()
    await svc.start()
    return svc


async def shutdown_sample_store_service() -> None:
    svc = get_sample_store_service()
    await svc.stop()


def get_async_sample_storage(profile_name: str, db_path: str) -> AsyncSampleStorage:
    return AsyncSampleStorage(get_sample_store_service(), profile_name=profile_name, db_path=db_path)
