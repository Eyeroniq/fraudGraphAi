from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from fraudgraph.data.download import (
    ChecksumMismatchError,
    MissingCredentialsError,
    fetch,
    kaggle_credentials,
)

CSV_NAME = "PS_test_log.csv"
CONTENT = b"step,type\n1,PAYMENT\n"
SHA = hashlib.sha256(CONTENT).hexdigest()


def make_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"nested/{CSV_NAME}", CONTENT)
    return path


@pytest.mark.parametrize("kind", ["zip", "csv"])
def test_fetch_from_local_file(tmp_path: Path, kind: str) -> None:
    if kind == "zip":
        source = make_zip(tmp_path / "dl.zip")
    else:
        source = tmp_path / "dl.csv"
        source.write_bytes(CONTENT)

    result = fetch(tmp_path / "raw", CSV_NAME, dataset="x/y", from_file=source, expected_sha256=SHA)

    assert result.path.read_bytes() == CONTENT
    assert result.verified
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == [CSV_NAME]  # no temp leftovers


def test_unpinned_checksum_is_reported_not_verified(tmp_path: Path) -> None:
    result = fetch(
        tmp_path / "raw",
        CSV_NAME,
        dataset="x/y",
        from_file=make_zip(tmp_path / "dl.zip"),
        expected_sha256=None,
    )

    assert (result.sha256, result.verified) == (SHA, False)


def test_checksum_mismatch_keeps_nothing(tmp_path: Path) -> None:
    with pytest.raises(ChecksumMismatchError):
        fetch(
            tmp_path / "raw",
            CSV_NAME,
            dataset="x/y",
            from_file=make_zip(tmp_path / "dl.zip"),
            expected_sha256="0" * 64,
        )

    assert not (tmp_path / "raw" / CSV_NAME).exists()


def test_existing_file_is_reverified_not_refetched(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    source = make_zip(tmp_path / "dl.zip")
    fetch(raw, CSV_NAME, dataset="x/y", from_file=source, expected_sha256=SHA)
    (raw / CSV_NAME).write_bytes(b"tampered\n")

    with pytest.raises(ChecksumMismatchError):
        fetch(raw, CSV_NAME, dataset="x/y", expected_sha256=SHA)


def test_zip_without_the_csv_fails(tmp_path: Path) -> None:
    archive = tmp_path / "dl.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("other.csv", CONTENT)

    with pytest.raises(FileNotFoundError, match=CSV_NAME):
        fetch(tmp_path / "raw", CSV_NAME, dataset="x/y", from_file=archive, expected_sha256=None)


def test_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    assert kaggle_credentials() == ("alice", "k")


def test_credentials_from_kaggle_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "bob", "key": "s"}))

    assert kaggle_credentials() == ("bob", "s")


def test_missing_credentials_explain_the_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))

    with pytest.raises(MissingCredentialsError, match="--from-file"):
        kaggle_credentials()
