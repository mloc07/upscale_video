"""CLI tool to execute commands on remote Google Colab via Cloudflare Bridge."""

from __future__ import annotations

import json
import sys
import urllib.request


def run_on_colab(bridge_url: str, command: str, cwd: str = "/content/upscale_video") -> int:
    endpoint = bridge_url.rstrip("/") + "/run"
    payload = json.dumps({"cmd": command, "cwd": cwd}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(data.get("output", ""))
            return data.get("exit_code", 0)
    except Exception as exc:
        print(f"[Error connecting to Colab Bridge]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/colab_bridge.py <BRIDGE_URL> '<COMMAND>' [CWD]")
        sys.exit(1)
    url = sys.argv[1]
    cmd = sys.argv[2]
    cwd = sys.argv[3] if len(sys.argv) > 3 else "/content/upscale_video"
    code = run_on_colab(url, cmd, cwd)
    sys.exit(code)
