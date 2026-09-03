"""Run from a project copy without installing SAGE as a package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from sage.phase1 import main
except ModuleNotFoundError as error:
    print(f"Missing dependency: {error.name}. See docs/Phase1/guide.md for setup.", file=sys.stderr)
    raise SystemExit(1)

if __name__ == "__main__":
    raise SystemExit(main())
