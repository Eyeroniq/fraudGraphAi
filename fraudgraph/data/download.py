"""Fetch the raw PaySim CSV into `data/raw/` and check its sha256.

Two sources:
- the Kaggle API, with `KAGGLE_USERNAME` / `KAGGLE_KEY` from the environment
  (or `~/.kaggle/kaggle.json`), or
- a zip or CSV the user downloaded by hand (`--from-file`).

The Kaggle credentials are sent only to kaggle.com, never forwarded on redirects
and never logged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/download/{dataset}"
CHUNK = 1 << 20


class ChecksumMismatchError(RuntimeError):
    """The raw file's sha256 differs from the pinned value."""


class MissingCredentialsError(RuntimeError):
    """No Kaggle credentials found."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    verified: bool  # False when no checksum is pinned yet


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str | None) -> DownloadResult:
    """Hash `path` and compare with `expected` (if pinned). Raises on mismatch."""
    actual = sha256_of(path)
    if expected is not None and actual != expected:
        raise ChecksumMismatchError(
            f"{path.name}: sha256 {actual} does not match pinned {expected}"
        )
    return DownloadResult(path=path, sha256=actual, verified=expected is not None)


def fetch(
    raw_dir: Path,
    csv_name: str,
    *,
    dataset: str,
    expected_sha256: str | None,
    from_file: Path | None = None,
    force: bool = False,
) -> DownloadResult:
    """Put `csv_name` into `raw_dir` and verify it. Skips work if it is already there
    (unless `force`). A file that fails the checksum is deleted, never kept."""
    target = raw_dir / csv_name
    if target.exists() and not force:
        return verify(target, expected_sha256)

    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=raw_dir, prefix=".download-") as tmp:
        tmp_dir = Path(tmp)
        if from_file is not None:
            source = from_file
        else:
            source = tmp_dir / "dataset.zip"
            _download_from_kaggle(dataset, source)
        staged = _extract_csv(source, csv_name, tmp_dir)
        try:
            result = verify(staged, expected_sha256)
        except ChecksumMismatchError:
            staged.unlink(missing_ok=True)
            raise
        os.replace(staged, target)
    return DownloadResult(path=target, sha256=result.sha256, verified=result.verified)


def _extract_csv(source: Path, csv_name: str, work_dir: Path) -> Path:
    """Return a CSV path inside `work_dir`, from a zip archive or a plain CSV."""
    staged = work_dir / csv_name
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            members = [m for m in zf.namelist() if Path(m).name == csv_name]
            if not members:
                raise FileNotFoundError(f"{csv_name} not found in {source.name}")
            with zf.open(members[0]) as src, staged.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
    else:
        shutil.copyfile(source, staged)
    return staged


def kaggle_credentials() -> tuple[str, str]:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if config.is_file():
        data = json.loads(config.read_text(encoding="utf-8"))
        if data.get("username") and data.get("key"):
            return str(data["username"]), str(data["key"])
    raise MissingCredentialsError(
        "Set KAGGLE_USERNAME and KAGGLE_KEY (or create ~/.kaggle/kaggle.json), "
        "or download the zip from Kaggle yourself and pass --from-file."
    )


def _download_from_kaggle(dataset: str, dest: Path) -> None:
    user, key = kaggle_credentials()
    request = urllib.request.Request(KAGGLE_API.format(dataset=dataset))
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    # Unredirected: the header is not forwarded when Kaggle redirects to storage.
    request.add_unredirected_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(request, timeout=300) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK)
