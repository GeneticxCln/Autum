import asyncio
import sys

from autum.app import main

try:
    sys.exit(asyncio.run(main()))
except KeyboardInterrupt:
    print("\nBye.")
