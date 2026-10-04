"""
Vercel Serverless Entrypoint for CycloneSense FastAPI Backend.
Exposes the FastAPI application instance for Vercel deployment and local execution.
"""
import sys
from pathlib import Path

# Ensure both backend directory and parent directory are on sys.path
backend_dir = Path(__file__).resolve().parent
repo_root = backend_dir.parent

for p in (str(repo_root), str(backend_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app.main import app

__all__ = ["app"]
