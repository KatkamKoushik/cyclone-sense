"""
Launcher script for CycloneSense V4 Streamlit Research Dashboard.
Executes Streamlit dashboard with automated port resolution and workspace environment.
"""

import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    dashboard_path = root / "dashboard" / "app.py"
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)

    print(f"============================================================")
    print(f"  CycloneSense V4 — Streamlit Research Dashboard Launcher   ")
    print(f"============================================================")
    print(f"• Dashboard entry point: {dashboard_path}")
    print(f"• Launching Streamlit on port 8501...")
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port=8501",
        "--server.headless=true",
        "--theme.base=dark",
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nCycloneSense Dashboard stopped by user.")

if __name__ == "__main__":
    main()
