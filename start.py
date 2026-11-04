#!/usr/bin/env python3
"""TruthLens AI - one-command launcher.

    python start.py              set up everything (first time) and run backend + frontend
    python start.py --setup      only install / update dependencies
    python start.py --no-browser run without opening the browser

First run: creates backend/.venv, installs Python packages, runs `npm install`.
Later runs: skip anything that is already installed (re-installs only if requirements change).
"""
import argparse
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV = BACKEND / ".venv"
IS_WINDOWS = os.name == "nt"
VENV_PYTHON = VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

BACKEND_PORT = 8000
FRONTEND_PORT = 5173
MIN_PY, MAX_PY = (3, 10), (3, 12)
MIN_NODE = 18


def say(msg: str) -> None:
    print(f"\033[96m[TruthLens]\033[0m {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"\n\033[91m[TruthLens] ERROR:\033[0m {msg}\n", flush=True)
    sys.exit(1)


def run(cmd: list, cwd: Path, what: str) -> None:
    say(f"{what} ...")
    if subprocess.call([str(c) for c in cmd], cwd=cwd) != 0:
        fail(f"{what} failed. Read the messages above, fix the problem and run start.py again.")


def file_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------ checks
def check_python() -> None:
    v = sys.version_info[:2]
    if v < MIN_PY:
        fail(f"Python {v[0]}.{v[1]} is too old. Install Python 3.11 or 3.12 from https://www.python.org/downloads/")
    if v > MAX_PY:
        say(f"WARNING: Python {v[0]}.{v[1]} is newer than tested (3.10-3.12). "
            "If installation fails, install Python 3.12 and run start.py with it.")


def find_npm() -> str:
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        fail("Node.js is not installed. Install the LTS version from https://nodejs.org/ and open a NEW terminal.")
    out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    try:
        major = int(out.lstrip("v").split(".")[0])
    except ValueError:
        major = 0
    if major < MIN_NODE:
        fail(f"Node.js {out} is too old. Install Node.js 20 or newer from https://nodejs.org/")
    return npm


def port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ------------------------------------------------------------------ setup
def setup_backend() -> None:
    if not VENV_PYTHON.exists():
        run([sys.executable, "-m", "venv", VENV], ROOT, "Creating Python virtual environment (backend/.venv)")

    req = BACKEND / "requirements.txt"
    marker = VENV / ".requirements.sha256"
    wanted = file_hash(req)
    if marker.exists() and marker.read_text().strip() == wanted:
        say("Python packages already installed.")
    else:
        run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"], BACKEND, "Upgrading pip")
        run([VENV_PYTHON, "-m", "pip", "install", "-r", req], BACKEND,
            "Installing Python packages (first time takes 5-15 minutes)")
        marker.write_text(wanted)

    env_file, example = BACKEND / ".env", BACKEND / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copy(example, env_file)
        say("Created backend/.env from .env.example")


def setup_frontend(npm: str) -> None:
    marker = FRONTEND / "node_modules" / ".package.sha256"
    wanted = file_hash(FRONTEND / "package.json", FRONTEND / "package-lock.json")
    if marker.exists() and marker.read_text().strip() == wanted:
        say("Frontend packages already installed.")
    else:
        run([npm, "install", "--no-fund", "--no-audit"], FRONTEND, "Installing frontend packages (npm install)")
        marker.write_text(file_hash(FRONTEND / "package.json", FRONTEND / "package-lock.json"))

    env_file, example = FRONTEND / ".env", FRONTEND / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copy(example, env_file)


# ------------------------------------------------------------------ run
def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_process(cmd: list, cwd: Path) -> subprocess.Popen:
    kwargs = {"cwd": cwd}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen([str(c) for c in cmd], **kwargs)


def wait_for(url: str, proc: subprocess.Popen, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(1)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up and run TruthLens AI")
    parser.add_argument("--setup", action="store_true", help="only install dependencies, do not start")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    os.system("")  # enables colours in old Windows terminals
    check_python()
    npm = find_npm()
    setup_backend()
    setup_frontend(npm)
    if args.setup:
        say("Setup complete. Run `python start.py` to start the app.")
        return

    for port, name in ((BACKEND_PORT, "backend"), (FRONTEND_PORT, "frontend")):
        if port_in_use(port):
            fail(f"Port {port} ({name}) is already in use. Close the other program (or an old TruthLens window) "
                 "and try again.")

    say(f"Starting backend  -> http://127.0.0.1:{BACKEND_PORT}  (API docs: /docs)")
    backend = start_process(
        [VENV_PYTHON, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", BACKEND_PORT], BACKEND)
    if not wait_for(f"http://127.0.0.1:{BACKEND_PORT}/api/health", backend, 120):
        stop(backend)
        fail("Backend did not start. Read the error messages above.")

    say(f"Starting frontend -> http://localhost:{FRONTEND_PORT}")
    frontend = start_process([npm, "run", "dev"], FRONTEND)
    if not wait_for(f"http://localhost:{FRONTEND_PORT}", frontend, 90):
        stop(frontend)
        stop(backend)
        fail("Frontend did not start. Read the error messages above.")

    url = f"http://localhost:{FRONTEND_PORT}"
    say(f"TruthLens AI is running at {url}   (press Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        while backend.poll() is None and frontend.poll() is None:
            time.sleep(1)
        say("A server stopped unexpectedly - shutting down.")
    except KeyboardInterrupt:
        say("Stopping ...")
    finally:
        stop(frontend)
        stop(backend)
        say("Stopped.")


if __name__ == "__main__":
    main()
