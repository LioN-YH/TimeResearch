from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.full.run_quitobench_full import main


if __name__ == "__main__":
    raise SystemExit(main(["--model", "patchtst", "--window", "96_48_S", "--master-port", "29517", *sys.argv[1:]]))
