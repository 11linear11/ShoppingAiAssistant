"""
Run all MCP servers.

Usage:
    python -m src.mcp_servers.run_servers

This starts all 3 MCP servers:
- Embedding Server (port 5003)
- Interpret Server (port 5004)
- Search Server (port 5002)
"""

import signal
import os
import subprocess
import sys
import time
from pathlib import Path


def run_servers():
    """Run all MCP servers as subprocesses."""
    
    python_exe = sys.executable
    project_root = Path(__file__).parent.parent.parent
    
    servers = [
        ("Embedding Server", "src.mcp_servers.embedding_server", 5003, "shopping-assistant-embedding"),
        ("Interpret Server", "src.mcp_servers.interpret_server", 5004, "shopping-assistant-interpret"),
        ("Search Server", "src.mcp_servers.search_server", 5002, "shopping-assistant-search"),
    ]
    
    processes = []
    running = True
    
    def shutdown(signum, frame):
        nonlocal running
        running = False
        print("\n� Shutting down servers...")
        for name, proc, _ in processes:
            proc.terminate()
        for name, proc, _ in processes:
            proc.wait()
        print("All servers stopped.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    print("�🚀 Starting MCP Servers...")
    print("=" * 50)
    
    for name, module, port, service_name in servers:
        print(f"Starting {name} on port {port}...")
        child_env = os.environ.copy()
        child_env.setdefault("PIPELINE_SERVICE_NAME", service_name)
        child_env.setdefault("LOGFIRE_SERVICE_NAME", service_name)
        if "DEBUG_LOG" not in child_env and "DEBUG_MODE" in child_env:
            child_env["DEBUG_LOG"] = child_env["DEBUG_MODE"]
        proc = subprocess.Popen(
            [python_exe, "-m", module],
            cwd=str(project_root),
            env=child_env,
        )
        processes.append((name, proc, port))
        print(f"  ✅ {name} started (PID: {proc.pid})")
        time.sleep(0.5)
    
    print("=" * 50)
    print("All servers started! Press Ctrl+C to stop.\n")
    print("Endpoints:")
    for name, _, port, _ in servers:
        print(f"  {name}: http://localhost:{port}/mcp")
    print()
    
    # Keep running
    while running:
        time.sleep(1)
        # Check if any died
        for i, (name, proc, port) in enumerate(processes):
            if proc.poll() is not None:
                print(f"⚠️ {name} exited, restarting...")
                child_env = os.environ.copy()
                child_env.setdefault("PIPELINE_SERVICE_NAME", servers[i][3])
                child_env.setdefault("LOGFIRE_SERVICE_NAME", servers[i][3])
                if "DEBUG_LOG" not in child_env and "DEBUG_MODE" in child_env:
                    child_env["DEBUG_LOG"] = child_env["DEBUG_MODE"]
                new_proc = subprocess.Popen([python_exe, "-m", servers[i][1]], cwd=str(project_root), env=child_env)
                processes[i] = (name, new_proc, port)


if __name__ == "__main__":
    run_servers()
