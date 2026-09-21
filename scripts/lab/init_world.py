#!/usr/bin/env python3
"""CLI utility to initialize a WorldModel database file."""

import asyncio
import sys
from pathlib import Path

from wow_bot.world.store import WorldModel


async def async_main(db_path: Path) -> int:
    """Initialize world database and print node count."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            return 0

        async with await WorldModel.open(db_path) as world:
            count = await world.count_nodes()
            print(count)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error initializing world database: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """Entry point for init_world script."""
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: init_world.py <db_path>", file=sys.stderr)
        return 1

    db_path = Path(sys.argv[1].strip())
    return asyncio.run(async_main(db_path))


if __name__ == "__main__":
    sys.exit(main())
