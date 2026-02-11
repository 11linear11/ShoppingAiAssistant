#!/usr/bin/env python3
"""
Shopping AI Assistant - Main Entry Point

A production-ready Persian shopping assistant using LangGraph
with multi-layer caching, semantic search, and conversational AI.

Usage:
    python main.py                  # Start interactive chat
    python main.py --servers        # Start MCP servers only
    python main.py --check          # Check service health
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


async def check_health():
    """Check health of all services."""
    import httpx

    services = [
        ("Embedding Server", "http://localhost:5003/health"),
        ("Interpret Server", "http://localhost:5004/health"),
        ("Search Server", "http://localhost:5002/health"),
    ]

    print("\n🔍 Checking service health...\n")

    async with httpx.AsyncClient(timeout=5.0) as client:
        all_healthy = True

        for name, url in services:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    print(f"✅ {name}: {status}")
                else:
                    print(f"⚠️  {name}: HTTP {response.status_code}")
                    all_healthy = False
            except httpx.ConnectError:
                print(f"❌ {name}: Not running")
                all_healthy = False
            except Exception as e:
                print(f"❌ {name}: {e}")
                all_healthy = False

    print()
    if all_healthy:
        print("✅ All services are healthy!")
    else:
        print("⚠️  Some services are not available.")
        print("Run 'python main.py --servers' to start them.")

    return all_healthy


async def run_chat():
    """Run the interactive chat interface."""
    from src.agent import create_agent, settings

    print("\n" + "=" * 60)
    print("🛒 Shopping AI Assistant (Persian)")
    print("=" * 60)
    print("\nدستیار خرید هوشمند فارسی")
    if settings.debug_mode:
        print("🔧 DEBUG MODE: Caching is DISABLED")
    print("برای خروج 'quit' یا 'خروج' تایپ کنید.\n")

    try:
        agent = await create_agent()
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {e}")
        print("\nMake sure all MCP servers are running:")
        print("  python main.py --servers")
        return

    session_id = None

    while True:
        try:
            # Get user input
            user_input = input("👤 شما: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "خروج", "q"]:
                print("\n👋 خداحافظ! امیدوارم کمکتون کرده باشم.")
                break

            # Process message
            print("\n⏳ در حال پردازش...")
            response, session_id = await agent.chat(user_input, session_id)

            print(f"\n🤖 دستیار:\n{response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 خداحافظ!")
            break
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            print(f"\n❌ خطا: {e}\n")

    await agent.close()


def start_servers():
    """Start all MCP servers."""
    from src.mcp_servers.run_servers import run_servers
    run_servers()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Shopping AI Assistant - Persian E-commerce Chatbot"
    )
    parser.add_argument(
        "--servers",
        action="store_true",
        help="Start MCP servers only",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check service health",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level)

    if args.servers:
        # Start MCP servers
        start_servers()
    elif args.check:
        # Check health
        asyncio.run(check_health())
    else:
        # Run chat interface
        asyncio.run(run_chat())


if __name__ == "__main__":
    main()
