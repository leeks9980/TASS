"""Download and verify the published MaleCNS v1.0 graph tables."""

import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
]
SUPPLEMENTARY = [
    "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "syn-points-male-cns-v1.0-minconf-0.5.feather",
    "syn-partners-male-cns-v1.0-minconf-0.5.feather",
    "tbar-neurotransmitters-male-cns-v1.0.feather",
]
DEST = Path(__file__).resolve().parent.parent / "data" / "male-cns-v1.0"


def download(name):
    DEST.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + name
    path = DEST / name
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        expected = int(response.headers["Content-Length"])
        hashes = response.headers.get("x-goog-hash", "")
        remote_md5 = next((v.strip()[4:] for v in hashes.split(",") if v.strip().startswith("md5=")), None)
        generation = response.headers.get("x-goog-generation")
    if not path.exists() or path.stat().st_size != expected:
        partial = path.with_suffix(path.suffix + ".part")
        downloaded = partial.stat().st_size if partial.exists() else 0
        if downloaded > expected:
            downloaded = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={downloaded}-"} if downloaded else {})
        response = urllib.request.urlopen(request, timeout=60)
        if downloaded and response.status != 206:
            downloaded = 0
        with response, partial.open("ab" if downloaded else "wb") as output:
            while block := response.read(4 * 1024 * 1024):
                output.write(block)
                downloaded += len(block)
                if downloaded // (64 * 1024 * 1024) != (downloaded - len(block)) // (64 * 1024 * 1024):
                    print(f"{name}: {downloaded / 1e6:.0f}/{expected / 1e6:.0f} MB", flush=True)
        if downloaded != expected:
            raise RuntimeError(f"Incomplete download: {name}: {downloaded} != {expected}")
        partial.replace(path)
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            md5.update(block)
            sha256.update(block)
    if remote_md5 and base64.b64encode(md5.digest()).decode() != remote_md5:
        raise RuntimeError(f"MD5 mismatch: {name}; remove the invalid file and rerun")
    print(f"Verified: {name} ({expected:,} bytes)", flush=True)
    return {"file": name, "url": url, "bytes": expected, "sha256": sha256.hexdigest(),
            "remote_md5": remote_md5, "generation": generation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-tables', action='store_true', help='Download all seven published flat-connectome tables (~24 GB).')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(download, FILES + SUPPLEMENTARY if args.all_tables else FILES))
    # A basic rerun must not erase metadata for already acquired supplementary files.
    previous = DEST / 'manifest.json'
    if previous.exists():
        old = json.loads(previous.read_text(encoding='utf-8'))
        names = {r['file'] for r in records}
        records.extend(r for r in old.get('files', []) if r['file'] not in names and (DEST / r['file']).exists())
    manifest = {"dataset": "male-cns:v1.0", "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "https://male-cns.janelia.org/download/", "license": "CC-BY", "files": records}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
