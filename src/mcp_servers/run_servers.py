#!/usr/bin/env python3
"""
MCP Servers Runner
Starts all MCP servers for the Shopping AI Assistant.

Servers:
- embedding_server (Port 5003) - Must start first
- interpret_server (Port 5004) - Depends on embedding_server
- equip_server (Port 5005) - EQuIP 3B DSL generator
- dsl_processor_server (Port 5006) - DSL processor, depends on embedding_server
- search_server (Port 5002) - Depends on embedding_server
"""

import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Server configurations
SERVERS = [
    {
        "name": "embedding-server",
        "port": 5003,
        "script": "embedding_server.py",
        "depends_on": [],
        "startup_delay": 3,  # Seconds to wait after starting
    },
    {
        "name": "interpret-server", 
        "port": 5004,
        "script": "interpret_server.py",
        "depends_on": ["embedding-server"],
        "startup_delay": 1,
    },
    {
        "name": "equip-server",
        "port": 5005,
        "script": "equip_server.py",
        "depends_on": [],
        "startup_delay": 1,
    },
    {
        "name": "dsl-processor-server",
        "port": 5006,
        "script": "dsl_processor_server.py",
        "depends_on": ["embedding-server"],
        "startup_delay": 1,
    },
    {
        "name": "search-server",
        "port": 5002,
        "script": "search_server.py",
        "depends_on": ["embedding-server"],
        "startup_delay": 1,
    },
]

# Global process list
processes = []


def get_server_path(script_name: str) -> Path:
    """Get the full path to a server script."""
    return Path(__file__).parent / script_name


def start_server(server_config: dict, verbose: bool = False) -> subprocess.Popen:
    """Start a single MCP server."""
    name = server_config["name"]
    port = server_config["port"]
    script = server_config["script"]
    script_path = get_server_path(script)
    
    if not script_path.exists():
        print(f"❌ Error: {script_path} not found!")
        return None
    
    print(f"🚀 Starting {name} on port {port}...")
    
    # Build command
    cmd = [sys.executable, str(script_path)]
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    # Start process
    if verbose:
        # Show output in terminal
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT)
        )
    else:
        # Redirect output to log files
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        
        stdout_log = open(log_dir / f"{name}.log", "w")
        stderr_log = open(log_dir / f"{name}.error.log", "w")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            stdout=stdout_log,
            stderr=stderr_log
        )
    
    return process


def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Wait for a server to be ready on the given port."""
    import socket
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def stop_all_servers():
    """Stop all running servers."""
    global processes
    print("\n🛑 Stopping all servers...")
    
    for process, name in processes:
        if process and process.poll() is None:
            print(f"   Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    processes = []
    print("✅ All servers stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    stop_all_servers()
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start MCP servers for Shopping AI Assistant")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show server output in terminal")
    parser.add_argument("--server", type=str, help="Start only a specific server (embedding, interpret, search)")
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("🛒 Shopping AI Assistant - MCP Servers")
    print("=" * 60)
    
    # Determine which servers to start
    if args.server:
        server_map = {
            "embedding": SERVERS[0],
            "interpret": SERVERS[1],
            "search": SERVERS[2],
        }
        if args.server not in server_map:
            print(f"❌ Unknown server: {args.server}")
            print(f"   Valid options: {', '.join(server_map.keys())}")
            sys.exit(1)
        servers_to_start = [server_map[args.server]]
    else:
        servers_to_start = SERVERS
    
    # Start servers
    global processes
    started_servers = set()
    
    for server in servers_to_start:
        # Check dependencies
        for dep in server["depends_on"]:
            if dep not in started_servers:
                print(f"⚠️  Warning: {server['name']} depends on {dep} which hasn't been started")
        
        # Start server
        process = start_server(server, verbose=args.verbose)
        if process:
            processes.append((process, server["name"]))
            
            # Wait for startup
            print(f"   Waiting {server['startup_delay']}s for {server['name']} to initialize...")
            time.sleep(server["startup_delay"])
            
            # Check if server is running
            if process.poll() is not None:
                print(f"❌ {server['name']} failed to start!")
                stop_all_servers()
                sys.exit(1)
            
            # Wait for port to be ready
            if wait_for_server(server["port"], timeout=10):
                print(f"✅ {server['name']} is ready on port {server['port']}")
                started_servers.add(server["name"])
            else:
                print(f"⚠️  {server['name']} started but port {server['port']} not responding yet")
                started_servers.add(server["name"])
    
    print()
    print("=" * 60)
    print("✅ All servers started successfully!")
    print("=" * 60)
    print()
    print("📡 Server endpoints:")
    for server in servers_to_start:
        print(f"   - {server['name']}: http://localhost:{server['port']}")
    print()
    print("📋 Logs are in: logs/")
    print("🛑 Press Ctrl+C to stop all servers")
    print()
    
    # Keep running until interrupted
    try:
        while True:
            # Check if any process has died
            for process, name in processes:
                if process.poll() is not None:
                    print(f"⚠️  {name} has stopped unexpectedly!")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    
    stop_all_servers()


if __name__ == "__main__":
    main()
