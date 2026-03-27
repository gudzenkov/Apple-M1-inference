from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Iterable, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI
from model_aliases import get_default_model_id, resolve_model_alias

DEFAULT_MODEL_ID = get_default_model_id()
DEFAULT_HOST = "0.0.0.0"
DEFAULT_START_TIMEOUT_SEC = 300


@dataclass(frozen=True)
class ServerControl:
    server_name: str
    script_path: Path
    pid_file: Path
    log_file: Path
    host: str
    port: int

    @property
    def root_dir(self) -> Path:
        return self.script_path.resolve().parents[2]

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/models"

    @property
    def shutdown_url(self) -> str:
        return f"http://{self.host}:{self.port}/internal/shutdown"


def resolve_runtime(default_port: int, port_env_name: str = "PORT") -> Tuple[str, str, int]:
    model_id = resolve_model_alias(os.getenv("HUGGINGFACE_MODEL", DEFAULT_MODEL_ID))
    host = os.getenv("HOST", DEFAULT_HOST)
    port_raw = os.getenv("PORT") or os.getenv(port_env_name) or str(default_port)
    port = int(port_raw)
    return model_id, host, port


def install_shutdown_endpoint(app: FastAPI, server_name: str) -> None:
    @app.post("/internal/shutdown")
    def shutdown() -> dict[str, str]:
        def _stop() -> None:
            time.sleep(0.2)
            signal.raise_signal(signal.SIGINT)

        threading.Thread(target=_stop, daemon=True).start()
        return {"status": "shutting_down", "server": server_name}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    content = pid_file.read_text(encoding="utf-8").strip()
    if not content.isdigit():
        return -1
    return int(content)


def _http_ok(url: str, method: str = "GET", timeout: float = 2.0) -> bool:
    req = Request(url=url, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:  # noqa: S310
            return 200 <= response.status < 300
    except URLError:
        return False


def _wait_for_health(url: str, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _http_ok(url, method="GET", timeout=2.0):
            return True
        time.sleep(1)
    return False


def _wait_for_exit(pid: int, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)
    return not _pid_alive(pid)


def _start(control: ServerControl) -> int:
    pid = _read_pid(control.pid_file)
    if pid is not None:
        if pid == -1:
            print(f"Invalid pid file: {control.pid_file}", file=sys.stderr)
            return 1
        if _pid_alive(pid):
            print(f"{control.server_name} already running (pid {pid})")
            return 0
        control.pid_file.unlink(missing_ok=True)

    control.log_file.parent.mkdir(parents=True, exist_ok=True)
    control.pid_file.parent.mkdir(parents=True, exist_ok=True)

    start_cmd = [sys.executable, str(control.script_path), "serve"]
    timeout_sec = int(os.getenv("SERVER_START_TIMEOUT", str(DEFAULT_START_TIMEOUT_SEC)))

    with open(control.log_file, "a", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(  # noqa: S603
            start_cmd,
            cwd=str(control.root_dir),
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    control.pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

    if _wait_for_health(control.health_url, timeout_sec=timeout_sec):
        print(
            f"{control.server_name} started on {control.host}:{control.port} "
            f"(pid {proc.pid})"
        )
        return 0

    print(
        f"{control.server_name} did not become ready within {timeout_sec}s. "
        f"Check logs: {control.log_file}",
        file=sys.stderr,
    )
    return 1


def _stop(control: ServerControl) -> int:
    pid = _read_pid(control.pid_file)
    if pid is None:
        print(f"{control.server_name} is not running (no pid file)")
        return 0
    if pid == -1:
        print(f"Invalid pid file: {control.pid_file}", file=sys.stderr)
        return 1

    if not _pid_alive(pid):
        control.pid_file.unlink(missing_ok=True)
        print(f"{control.server_name} already stopped (stale pid {pid})")
        return 0

    if not _http_ok(control.shutdown_url, method="POST", timeout=3.0):
        print(
            f"Shutdown request failed for {control.server_name}: {control.shutdown_url}",
            file=sys.stderr,
        )
        return 1

    if not _wait_for_exit(pid, timeout_sec=20.0):
        print(
            f"Timed out waiting for {control.server_name} to stop (pid {pid})",
            file=sys.stderr,
        )
        return 1

    control.pid_file.unlink(missing_ok=True)
    print(f"{control.server_name} stopped (pid {pid})")
    return 0


def _status(control: ServerControl) -> int:
    pid = _read_pid(control.pid_file)
    if pid is None:
        print(f"{control.server_name}: stopped (no pid file)")
        return 0
    if pid == -1:
        print(f"{control.server_name}: invalid pid file {control.pid_file}", file=sys.stderr)
        return 1
    alive = _pid_alive(pid)
    healthy = _http_ok(control.health_url, method="GET", timeout=2.0)
    state = "running" if alive else "stale-pid"
    print(
        f"{control.server_name}: {state}; pid={pid}; "
        f"health={'ok' if healthy else 'down'}; endpoint={control.health_url}"
    )
    return 0 if alive else 1


def handle_management_command(control: ServerControl, argv: Iterable[str]) -> Optional[int]:
    args = list(argv)
    if not args:
        return None

    cmd = args[0]
    if cmd == "serve":
        return None
    if cmd == "start":
        return _start(control)
    if cmd == "stop":
        return _stop(control)
    if cmd == "status":
        return _status(control)

    print(
        f"Unknown command '{cmd}'. Use one of: serve, start, stop, status.",
        file=sys.stderr,
    )
    return 2
