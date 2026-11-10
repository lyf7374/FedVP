"""Fetch the pinned FedCoT data processors and prompt from their upstream repository."""
import argparse
from pathlib import Path
import urllib.request

REVISION = "5956c1e6c6be2c2b0282b8d7869c282da61e5909"
FILES = ("src/dataset_loader.py", "src/prompt_loader.py")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for relative in FILES:
        path = args.root / "upstream/FedCoT" / relative
        if path.exists():
            print(f"Already present: {path}")
            continue
        url = f"https://raw.githubusercontent.com/DIaacKr/FedCoT/{REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"Fetched {relative} from FedCoT revision {REVISION}")

if __name__ == "__main__":
    main()
