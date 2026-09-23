"""Fetch Restormer motion-deblurring weights into backend/ai/weights.

Per the PixRevive runtime rules: downloads use urllib.request against GitHub
releases; weights live under backend/ai/weights/, never a user cache.
"""

import os
import sys
import urllib.request

URL = "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth"
OUT = os.path.join("backend", "ai", "weights", "restormer", "motion_deblurring.pth")


def main() -> int:
    if os.path.exists(OUT) and os.path.getsize(OUT) > 90 * 1024 * 1024:
        print(f"[OK] already present: {OUT} ({os.path.getsize(OUT)} bytes)")
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"Downloading {URL}")

    def _hook(count, block, total):
        done = count * block
        if total > 0 and count % 200 == 0:
            print(f"  {done / 1e6:7.1f} MB / {total / 1e6:.1f} MB", flush=True)

    try:
        urllib.request.urlretrieve(URL, OUT, _hook)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1
    size = os.path.getsize(OUT)
    print(f"[OK] wrote {OUT} ({size} bytes)")
    if size < 90 * 1024 * 1024:
        print("[WARN] size below expected ~100MB — verify the file")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
