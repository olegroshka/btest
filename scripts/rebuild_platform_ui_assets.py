from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    frontend = repo / "frontend"

    subprocess.check_call(["npm", "run", "build"], cwd=str(frontend))


if __name__ == "__main__":
    main()

