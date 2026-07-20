import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


project_root = Path(__file__).resolve().parent
ui_folder = project_root / "haikyuu_chat_ui"
api_host = "127.0.0.1"
api_port = 8001
ui_port = 5173


def print_step(message):
    print(f"[haikyuu-chat] {message}", flush = True)


def command_exists(command):
    return shutil.which(command) is not None


def run_checked(command, cwd, timeout = None):
    print_step(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd = cwd, check = True, timeout = timeout)


def start_process(name, command, cwd, env = None):
    print_step(f"Starting {name}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd = cwd,
        env = env,
        stdout = None,
        stderr = None,
        stdin = None,
    )


def stop_process(process, name):
    if process is None or process.poll() is not None:
        return

    print_step(f"Stopping {name}...")
    process.terminate()
    try:
        process.wait(timeout = 10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout = 5)


def install_ui_dependencies():
    if not command_exists("npm"):
        raise RuntimeError(
            "npm was not found. Install Node.js/npm first, then run this command again. "
            "Inside Conda, you can use: conda install -c conda-forge nodejs"
        )

    node_modules = ui_folder / "node_modules"
    if node_modules.exists():
        print_step("Frontend dependencies already installed.")
        return

    print_step("Frontend dependencies are missing; installing them now. This can take a few minutes.")
    try:
        run_checked(["npm", "install", "--loglevel", "warn"], cwd = ui_folder, timeout = 240)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "npm install did not finish within 240 seconds. Run this manually to see npm's full output: "
            "cd /home/serdoriko/projects/chatbot/haikyuu_chat_ui && npm install"
        ) from error


def main():
    parser = argparse.ArgumentParser(description = "Run the Haikyuu RAG API and React chat UI together.")
    parser.add_argument("--skip-install", action = "store_true", help = "Do not run npm install if node_modules is missing.")
    parser.add_argument("--api-port", type = int, default = api_port, help = "Port for the FastAPI backend.")
    parser.add_argument("--ui-port", type = int, default = ui_port, help = "Port for the Vite React UI.")
    args = parser.parse_args()

    if not ui_folder.exists():
        raise RuntimeError(f"React UI folder not found: {ui_folder}")

    api_process = None
    ui_process = None

    env = os.environ.copy()
    env["VITE_HAIKYUU_API_URL"] = f"http://{api_host}:{args.api_port}"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        print_step("The backend output is not hidden; Unsloth and Uvicorn messages will print below.")
        api_process = start_process(
            "streaming RAG API",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "haikyuu_rag_chat_api:app",
                "--host",
                api_host,
                "--port",
                str(args.api_port),
            ],
            cwd = project_root,
        )

        time.sleep(2)
        if api_process.poll() is not None:
            raise RuntimeError(f"Backend exited with code {api_process.returncode}.")

        if not args.skip_install:
            install_ui_dependencies()
        elif not command_exists("npm"):
            raise RuntimeError("npm was not found. Install Node.js/npm or remove --skip-install.")

        ui_process = start_process(
            "React chat UI",
            ["npm", "run", "dev", "--", "--host", api_host, "--port", str(args.ui_port)],
            cwd = ui_folder,
            env = env,
        )

        print_step(f"Backend: http://{api_host}:{args.api_port}")
        print_step(f"Frontend: http://{api_host}:{args.ui_port}")
        print_step("Press Ctrl+C to stop both processes.")

        while True:
            time.sleep(1)
            if api_process.poll() is not None:
                raise RuntimeError(f"Backend exited with code {api_process.returncode}.")
            if ui_process.poll() is not None:
                raise RuntimeError(f"Frontend exited with code {ui_process.returncode}.")
    except KeyboardInterrupt:
        print_step("Shutdown requested.")
    finally:
        stop_process(ui_process, "React chat UI")
        stop_process(api_process, "streaming RAG API")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        main()
    except Exception as error:
        print_step(str(error))
        sys.exit(1)
