"""Package module entrypoint (`python -m news_bot`)."""

from __future__ import annotations

import sys

from news_bot.cli import main

if __name__ == "__main__":
    sys.exit(main())

