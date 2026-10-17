"""CLI tool to execute commands and scripts on remote Google Colab via Cloudflare Bridge."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_on_colab(bridge_url: str, command: str, cwd: str = "/content/upscale_video", timeout: int = 900) -> int:
    endpoint = bridge_url.rstrip("/") + "/run"
    payload = json.dumps({"cmd": command, "cwd": cwd}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(data.get("output", ""))
            return data.get("exit_code", 0)
    except Exception as exc:
        print(f"[Error connecting to Colab Bridge]: {exc}", file=sys.stderr)
        return 1


def upload_to_colab(bridge_url: str, local_path: str | Path, remote_path: str) -> bool:
    endpoint = bridge_url.rstrip("/") + "/write"
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"File not found: {local_path}", file=sys.stderr)
        return False
    data_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    payload = json.dumps({"path": remote_path, "data_b64": data_b64, "append": False}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return "size" in res
    except Exception as exc:
        print(f"[Upload error]: {exc}", file=sys.stderr)
        return False


def run_script_on_colab(bridge_url: str, local_py_file: str | Path, cwd: str = "/content/upscale_video", timeout: int = 900) -> int:
    local_path = Path(local_py_file)
    remote_tmp = f"/tmp/{local_path.name}"
    if not upload_to_colab(bridge_url, local_path, remote_tmp):
        return 1
    return run_on_colab(bridge_url, f"python3 {remote_tmp}", cwd=cwd, timeout=timeout)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python scripts/colab_bridge.py <BRIDGE_URL> '<COMMAND>' [CWD]")
        print("  python scripts/colab_bridge.py <BRIDGE_URL> --script <LOCAL_PY_FILE> [CWD]")
        print("  python scripts/colab_bridge.py <BRIDGE_URL> --upload <LOCAL_FILE> <REMOTE_PATH>")
        sys.exit(1)

    url = sys.argv[1]
    if sys.argv[2] == "--script":
        py_file = sys.argv[3]
        cwd = sys.argv[4] if len(sys.argv) > 4 else "/content/upscale_video"
        sys.exit(run_script_on_colab(url, py_file, cwd))
    elif sys.argv[2] == "--upload":
        loc = sys.argv[3]
        rem = sys.argv[4]
        ok = upload_to_colab(url, loc, rem)
        sys.exit(0 if ok else 1)
    else:
        cmd = sys.argv[2]
        cwd = sys.argv[3] if len(sys.argv) > 3 else "/content/upscale_video"
        sys.exit(run_on_colab(url, cmd, cwd))
