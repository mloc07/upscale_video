"""Script to fetch benchmark results and videos from Colab bridge."""

import base64
import json
from pathlib import Path
import sys
import urllib.request


def download_file(bridge_url: str, remote_path: str, local_path: Path):
    cmd = f"python3 -c \"import base64; print(base64.b64encode(open('{remote_path}', 'rb').read()).decode('ascii'))\""
    endpoint = bridge_url.rstrip("/") + "/run"
    payload = json.dumps({"cmd": cmd, "cwd": "/content/upscale_video"}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Antigravity/ColabBridge"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if data.get("exit_code") != 0:
            print(f"Error fetching {remote_path}: {data.get('output')}")
            return False
        b64_str = data.get("output", "").strip()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(base64.b64decode(b64_str))
        print(f"Downloaded {remote_path} -> {local_path} ({local_path.stat().st_size} bytes)")
        return True


if __name__ == "__main__":
    bridge_url = sys.argv[1] if len(sys.argv) > 1 else "https://steve-nylon-win-asn.trycloudflare.com"
    out_dir = Path("eval/results/seedvr2")

    files = [
        "benchmark_metrics.json",
        "metrics_clip_A.json",
        "metrics_clip_B.json",
        "metrics_clip_C.json",
        "clip_A_seedvr2.mp4",
        "clip_B_seedvr2.mp4",
        "clip_C_seedvr2.mp4",
    ]

    for fname in files:
        rem = f"/content/upscale_video/eval/results/seedvr2/{fname}"
        loc = out_dir / fname
        download_file(bridge_url, rem, loc)
