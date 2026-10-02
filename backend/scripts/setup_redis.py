"""
Download and launch portable Redis server on Windows for CycloneSense.
"""
import os
import sys
import zipfile
import urllib.request
import subprocess
from pathlib import Path

REDIS_VERSION = "5.0.14.1"
REDIS_URL = f"https://github.com/tporadowski/redis/releases/download/v{REDIS_VERSION}/Redis-x64-{REDIS_VERSION}.zip"
DEST_DIR = Path("d:/CycloneSense/tools/redis")


def setup_redis():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    server_exe = DEST_DIR / "redis-server.exe"

    if not server_exe.exists():
        zip_path = DEST_DIR / "redis.zip"
        print(f"Downloading Redis v{REDIS_VERSION} from {REDIS_URL} ...")
        urllib.request.urlretrieve(REDIS_URL, zip_path)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DEST_DIR)
        if zip_path.exists():
            zip_path.unlink()
        print(f"Redis extracted to {DEST_DIR}")
    else:
        print(f"Redis is already downloaded at {server_exe}")

    print("Verifying files:")
    for f in DEST_DIR.glob("*.exe"):
        print(f" - {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    setup_redis()
