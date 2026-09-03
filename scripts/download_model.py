"""Fetch the pinned public VOC detector, retaining source and checksum records."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


REVISION = "bb17b6c3eef36d80be441ae8e5339be66e8e3b7a"
BASE = f"https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/{REVISION}"
FILES = ("deploy.prototxt", "mobilenet_iter_73000.caffemodel", "LICENSE")
EXPECTED_SHA256 = {
    "deploy.prototxt": "2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608",
    "mobilenet_iter_73000.caffemodel": "52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7",
    "LICENSE": "5de433821cfe672af2fca73c3005f48af16121c95f6a11e1031b07807ea59905",
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "mobilenet-ssd")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "https://github.com/chuanqi305/MobileNet-SSD",
                "revision": REVISION, "dataset": "VOC0712", "files": {}}
    for filename in FILES:
        target = args.output / filename
        url = f"{BASE}/{filename}"
        # No silent reuse of unknown weights: obtain the pinned revision each run.
        temporary = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Empty download: {url}")
            if digest.hexdigest() != EXPECTED_SHA256[filename]:
                raise RuntimeError(f"Checksum mismatch: {filename}; existing file was preserved")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        manifest["files"][filename] = {"url": url, "sha256": digest.hexdigest(),
                                      "bytes": target.stat().st_size}
        print(f"Downloaded {filename} ({target.stat().st_size:,} bytes)")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Model and provenance saved in {args.output}")


if __name__ == "__main__":
    main()
