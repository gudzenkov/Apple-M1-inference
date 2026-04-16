from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import requests


def _reasoning_payload(reasoning_mode: str) -> dict[str, str] | None:
    mode = str(reasoning_mode or "off").strip().lower()
    if mode == "off":
        return {"effort": "none"}
    if mode == "on":
        return {"effort": "low"}
    return None


def pids_for_listen_port(port: int) -> Set[int]:
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-tiTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return set()

    pids: Set[int] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def pids_for_pattern(pattern: str) -> Set[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return set()

    pids: Set[int] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_pid_exit(pid: int, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.2)
    return not pid_alive(pid)


def stop_pid_gracefully(pid: int) -> bool:
    if not pid_alive(pid):
        return True

    for sig, timeout in ((signal.SIGINT, 10.0), (signal.SIGTERM, 10.0)):
        if not pid_alive(pid):
            return True
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if wait_for_pid_exit(pid, timeout):
            return True

    return not pid_alive(pid)


def stop_mlx_servers(verbose: bool = True) -> None:
    pids: Set[int] = set()
    pids |= pids_for_listen_port(8000)
    pids |= pids_for_listen_port(8080)
    pids |= pids_for_pattern("mlx-openai-server.py")
    pids |= pids_for_pattern("mlx-openai-optiq-server.py")
    pids |= pids_for_pattern("mlx_openai_server")
    pids |= pids_for_pattern("mlx_openai_optiq_server")
    pids |= pids_for_pattern("mlx-openai-server")
    pids |= pids_for_pattern("mlx-openai-optiq-server")

    if not pids:
        return

    if verbose:
        print(f"Stopping existing MLX server processes: {sorted(pids)}", file=sys.stderr)

    failed: List[int] = []
    for pid in sorted(pids):
        if not stop_pid_gracefully(pid):
            failed.append(pid)

    if failed:
        raise RuntimeError(f"Failed to stop running MLX server processes: {failed}")

    if pids_for_listen_port(8000) or pids_for_listen_port(8080):
        raise RuntimeError("Ports 8000/8080 are still occupied after stop attempts")


def stop_llama_cpp_servers(port: int, verbose: bool = True) -> None:
    pids: Set[int] = set()
    pids |= pids_for_listen_port(port)
    pids |= pids_for_pattern(rf"llama-server.*--port[ =]{port}")

    if not pids:
        return

    if verbose:
        print(f"Stopping existing llama.cpp server processes on port {port}: {sorted(pids)}", file=sys.stderr)

    failed: List[int] = []
    for pid in sorted(pids):
        if not stop_pid_gracefully(pid):
            failed.append(pid)

    if failed:
        raise RuntimeError(f"Failed to stop running llama.cpp server processes: {failed}")

    if pids_for_listen_port(port):
        raise RuntimeError(f"Port {port} is still occupied after stop attempts")


def wait_for_server_ready(health_url: str, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _render_start_cmd(start_cmd: List[str], *, model: str, port: int) -> List[str]:
    host = "127.0.0.1"
    variables = {
        "model": model,
        "host": host,
        "port": str(port),
    }
    rendered: List[str] = []
    for arg in start_cmd:
        try:
            rendered.append(arg.format_map(variables))
        except KeyError as exc:
            raise RuntimeError(f"Unknown placeholder in managed server command: {exc}") from exc
    return rendered


def start_managed_server(
    runtime: str,
    model: str,
    timeout_sec: int,
    root_dir: Path,
    config: Dict[str, Any],
    log_dir: Path,
) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = root_dir / config["log_file"]
    log_handle = open(log_path, "a", encoding="utf-8")

    env = os.environ.copy()
    env["HUGGINGFACE_MODEL"] = model
    env["MODEL_ID"] = model
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(config["port"])
    start_cmd = _render_start_cmd(config["start_cmd"], model=model, port=int(config["port"]))

    print(f"Starting {runtime} server for model {model}...", file=sys.stderr)
    proc = subprocess.Popen(
        start_cmd,
        cwd=str(root_dir),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._log_handle = log_handle  # type: ignore[attr-defined]

    if wait_for_server_ready(config["health_url"], timeout_sec=timeout_sec):
        print(f"✓ {runtime} server ready on port {config['port']}", file=sys.stderr)
        return proc

    stop_managed_process(proc)
    raise RuntimeError(
        f"{runtime} server did not become ready within {timeout_sec}s. "
        f"Check logs: {log_path}"
    )


def stop_managed_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    log_handle = getattr(proc, "_log_handle", None)
    if log_handle is not None and not log_handle.closed:
        log_handle.close()


def warmup_model(chat_url: str, model: str, reasoning_mode: str = "off") -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "stream": False,
    }
    reasoning_payload = _reasoning_payload(reasoning_mode)
    if reasoning_payload is not None:
        payload["reasoning"] = reasoning_payload

    print(f"Warming up {model}...", file=sys.stderr)
    started_at = time.perf_counter()
    try:
        response = requests.post(chat_url, json=payload, timeout=180)
        warmup_sec = time.perf_counter() - started_at
        if response.status_code == 200:
            print("✓ Model warmed up", file=sys.stderr)
            return {
                "success": True,
                "warmup_sec": round(warmup_sec, 4),
                "status_code": response.status_code,
            }
        else:
            print(f"⚠ Warmup returned HTTP {response.status_code}", file=sys.stderr)
            return {
                "success": False,
                "warmup_sec": round(warmup_sec, 4),
                "status_code": response.status_code,
                "error": f"HTTP {response.status_code}",
            }
    except Exception as exc:  # noqa: BLE001
        warmup_sec = time.perf_counter() - started_at
        print(f"⚠ Warmup failed: {exc}", file=sys.stderr)
        return {
            "success": False,
            "warmup_sec": round(warmup_sec, 4),
            "error": str(exc),
        }
