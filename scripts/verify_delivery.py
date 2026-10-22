#!/usr/bin/env python3
"""Check the saved DeepSilencer reading delivery; no model computations."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import zipfile


def verify(root):
    manifest = json.loads((root / "verification/delivery-manifest.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            failures.append(f"Missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"Hash differs: {name}")
    package = root / "dist/DeepSilencer-agent/skill/deepsilencer-paper"
    for name in ("SKILL.md", "references/index.md", "references/paper.md", "references/supplement.md"):
        if not (package / name).is_file():
            failures.append(f"Missing skill component: {name}")
    for md in package.rglob("*.md"):
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", md.read_text()):
            target = urlsplit(link.strip("<>"))
            if target.scheme or target.netloc or not target.path:
                continue
            if not (md.parent / unquote(target.path)).exists():
                failures.append(f"Broken link in {md.relative_to(root)}: {link}")
    for name, expected_rows in manifest["csv_rows_including_header"].items():
        rows = list(csv.reader(io.StringIO((root / name).read_text(encoding="utf-8-sig"))))
        if len(rows) != expected_rows or not rows or rows[0] != ["siRNA", "mRNA", "label", "y", "td"] or any(len(r) != 5 for r in rows):
            failures.append(f"CSV structure differs: {name}")
    archive_path = root / manifest["archive"]
    with zipfile.ZipFile(archive_path) as archive:
        members = {n for n in archive.namelist() if not n.endswith("/")}
        expected_members = {
            str(p.relative_to(root / "dist")): p
            for p in (root / "dist/DeepSilencer-agent").rglob("*") if p.is_file()
        }
        if members != set(expected_members):
            failures.append("Archive inventory differs from delivered folder")
        for name in members & set(expected_members):
            if archive.read(name) != expected_members[name].read_bytes():
                failures.append(f"Archive bytes differ: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(manifest['files'])} file hashes, local skill links, three complete CSVs, and archive bytes.")
    print("Scope: reading-package integrity. DeepSilencer model execution remains unavailable.")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
