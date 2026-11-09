"""Download helpers shared by the dataset download commands."""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

USER_AGENT = "clean-sweep-data/0.1"


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    """Download url to dest, skipping files that already exist. Writes atomically."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
            partial.replace(dest)
            return
        except OSError:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise


def fetch_many(jobs: list[tuple[str, Path]], workers: int = 8,
               after: Callable[[Path], None] | None = None) -> list[str]:
    """Download (url, dest) pairs in parallel, calling `after(dest)` on each finished file.

    Returns the URLs that failed.
    """
    def job(url: str, dest: Path) -> None:
        fetch(url, dest)
        if after:
            after(dest)

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, url, dest): url for url, dest in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            if future.exception() is not None:
                failed.append(futures[future])
            if done % 100 == 0 or done == len(jobs):
                sys.stdout.write(f"\r  {done}/{len(jobs)} files ({len(failed)} failed)")
                sys.stdout.flush()
    print()
    return failed


def extract_zip(archive: Path, dest: Path) -> None:
    """Extract, refusing entries that would land outside dest."""
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.filename.startswith("__MACOSX/"):
                continue
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"unsafe path in {archive.name}: {member.filename}")
            zf.extract(member, dest)
