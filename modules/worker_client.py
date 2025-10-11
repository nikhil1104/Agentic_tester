# modules/worker_client.py
from __future__ import annotations
from typing import Dict, Any, Optional
import time
import math
import httpx


class WorkerClient:
    """
    Minimal sync client for the worker /v1 API.
    - Uses X-API-KEY if provided
    - Supports idempotency via X-Idempotency-Key
    - Simple exponential backoff on transient failures (no extra deps)
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        headers = {"X-API-KEY": api_key} if api_key else {}
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    # --- context manager sugar ---
    def __enter__(self) -> "WorkerClient":
        return self
    def __exit__(self, *exc):
        self.close()

    # --- simple retry/backoff wrapper ---
    def _request_with_retries(self, method: str, url: str, max_attempts: int = 3, **kwargs) -> httpx.Response:
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < max_attempts:
            try:
                r = self.client.request(method, url, **kwargs)
                # Retry on 5xx
                if 500 <= r.status_code < 600:
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                return r
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                attempt += 1
                if attempt >= max_attempts:
                    raise
                sleep = min(30.0, 1.0 * math.pow(2, attempt - 1))
                time.sleep(sleep)

        # should not reach
        assert last_exc is not None
        raise last_exc

    def submit(
        self,
        stage: str,
        plan: Dict[str, Any],
        execution_id: str,
        reports_dir: str,
        timeout_sec: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> str:
        payload = {
            "stage": stage,
            "plan": plan,
            "execution_id": execution_id,
            "reports_dir": reports_dir,
            "timeout_sec": timeout_sec,
        }
        headers = {}
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key
        r = self._request_with_retries("POST", "/v1/run", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["job_id"]

    def status(self, job_id: str) -> Dict[str, Any]:
        r = self._request_with_retries("GET", f"/v1/status/{job_id}")
        r.raise_for_status()
        return r.json()

    def result(self, job_id: str) -> Dict[str, Any]:
        r = self._request_with_retries("GET", f"/v1/result/{job_id}")
        r.raise_for_status()
        return r.json()

    def wait(self, job_id: str, poll_interval: float = 1.0, overall_timeout: Optional[int] = None) -> Dict[str, Any]:
        start = time.time()
        while True:
            s = self.status(job_id)
            st = s["status"]
            if st in ("done", "failed", "cancelled"):
                return self.result(job_id)
            if overall_timeout and (time.time() - start) > overall_timeout:
                raise TimeoutError("worker wait timeout")
            time.sleep(poll_interval)

    def cancel(self, job_id: str) -> Dict[str, Any]:
        r = self._request_with_retries("POST", f"/v1/cancel/{job_id}")
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


# --- Optional async variant for future async Runner integrations ---
class AsyncWorkerClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 20.0):
        headers = {"X-API-KEY": api_key} if api_key else {}
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    async def __aenter__(self) -> "AsyncWorkerClient":
        return self
    async def __aexit__(self, *exc):
        await self.client.aclose()

    async def submit(self, stage: str, plan: Dict[str, Any], execution_id: str, reports_dir: str, timeout_sec: Optional[int] = None, idempotency_key: Optional[str] = None) -> str:
        payload = {"stage": stage, "plan": plan, "execution_id": execution_id, "reports_dir": reports_dir, "timeout_sec": timeout_sec}
        headers = {"X-Idempotency-Key": idempotency_key} if idempotency_key else None
        r = await self.client.post("/v1/run", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["job_id"]

    async def status(self, job_id: str) -> Dict[str, Any]:
        r = await self.client.get(f"/v1/status/{job_id}")
        r.raise_for_status()
        return r.json()

    async def result(self, job_id: str) -> Dict[str, Any]:
        r = await self.client.get(f"/v1/result/{job_id}")
        r.raise_for_status()
        return r.json()

    async def wait(self, job_id: str, poll_interval: float = 1.0, overall_timeout: Optional[int] = None) -> Dict[str, Any]:
        start = time.time()
        while True:
            s = await self.status(job_id)
            if s["status"] in ("done", "failed", "cancelled"):
                return await self.result(job_id)
            if overall_timeout and (time.time() - start) > overall_timeout:
                raise TimeoutError("worker wait timeout")
            await asyncio.sleep(poll_interval)

    async def cancel(self, job_id: str) -> Dict[str, Any]:
        r = await self.client.post(f"/v1/cancel/{job_id}")
        r.raise_for_status()
        return r.json()
