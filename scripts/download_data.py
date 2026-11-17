"""Download and extract Spider and BIRD dev.

Sources (verified 2026-11-16):
  Spider: official release linked from https://yale-lily.github.io/spider
          (Google Drive id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J, 206 MB zip, CC BY-SA 4.0)
  BIRD:   https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
          linked from https://bird-bench.github.io/

Usage:
  python scripts/download_data.py spider
  python scripts/download_data.py bird

Layout produced:
  data/spider/{train_spider.json,dev.json,tables.json,database/<db_id>/<db_id>.sqlite}
  data/bird/{dev.json,dev_databases/<db_id>/<db_id>.sqlite}

Spider's test_database/ (865 MB) is the hidden-test release and is not used by
this project, so it is not extracted.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

SPIDER_KEEP = ("database/", "train_spider.json", "dev.json", "tables.json", "README.txt")


def _extract(z: zipfile.ZipFile, member: zipfile.ZipInfo, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with z.open(member) as src, open(dst, "wb") as out:
        shutil.copyfileobj(src, out, length=1 << 20)


def spider() -> None:
    zpath = RAW / "spider_data.zip"
    if not zpath.exists():
        import gdown
        RAW.mkdir(parents=True, exist_ok=True)
        gdown.download(id=SPIDER_GDRIVE_ID, output=str(zpath), quiet=False)
    out = ROOT / "data" / "spider"
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            name = m.filename
            if m.is_dir() or "__MACOSX" in name or name.endswith(".DS_Store"):
                continue
            rel = name.split("/", 1)[1] if "/" in name else name
            if not rel.startswith(SPIDER_KEEP):
                continue
            _extract(z, m, out / rel)
            n += 1
    print(f"spider: extracted {n} files to {out}")


def bird() -> None:
    zpath = RAW / "bird_dev.zip"
    if not zpath.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BIRD_DEV_URL}")
        urllib.request.urlretrieve(BIRD_DEV_URL, zpath)
    out = ROOT / "data" / "bird"
    inner = None
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            name = m.filename
            if m.is_dir() or "__MACOSX" in name or name.endswith(".DS_Store"):
                continue
            base = name.rsplit("/", 1)[-1]
            if base in ("dev.json", "dev.sql", "dev_tables.json"):
                _extract(z, m, out / base)
            elif base == "dev_databases.zip":
                inner = RAW / "bird_dev_databases.zip"
                _extract(z, m, inner)
    if inner is None:
        raise RuntimeError("dev_databases.zip not found inside BIRD dev.zip")
    n = 0
    with zipfile.ZipFile(inner) as z:
        for m in z.infolist():
            name = m.filename
            if m.is_dir() or "__MACOSX" in name or not name.endswith(".sqlite"):
                continue
            db_id = Path(name).stem
            _extract(z, m, out / "dev_databases" / db_id / f"{db_id}.sqlite")
            n += 1
    inner.unlink()
    print(f"bird: extracted {n} databases to {out}")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["spider", "bird"]
    for t in targets:
        {"spider": spider, "bird": bird}[t]()
