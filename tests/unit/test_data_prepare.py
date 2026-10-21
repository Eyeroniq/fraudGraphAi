from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from fraudgraph.data.download import ChecksumMismatchError
from fraudgraph.data.prepare import DataValidationError, prepare

HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"
)
# Deliberately out of step order, with two rows sharing step 2, to test the sort.
ROWS = [
    "2,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0",
    "1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0",
    "2,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0",
    "1,DEBIT,5337.77,C712410124,41720.0,36382.23,C195600860,41898.0,40348.79,0,0",
    "3,CASH_IN,229133.94,C905080434,15325.0,244458.94,C476402209,5083.0,51513.44,0,0",
]


def write_csv(path: Path, rows: list[str], header: str = HEADER) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    return write_csv(tmp_path / "paysim.csv", ROWS)


def test_prepare_writes_canonical_sorted_parquet(raw_csv: Path, tmp_path: Path) -> None:
    parquet, meta, written = prepare(raw_csv, tmp_path / "out")

    frame = pl.read_parquet(parquet)
    assert written
    assert frame.columns[:3] == ["txn_id", "step", "type"]
    assert "name_orig" in frame.columns
    assert "nameOrig" not in frame.columns
    assert frame["step"].to_list() == [1, 1, 2, 2, 3]
    # txn_id follows raw file order; ties in step keep that order.
    assert frame["txn_id"].to_list() == ["T0000001", "T0000003", "T0000000", "T0000002", "T0000004"]
    assert frame.schema["is_fraud"] == pl.Boolean
    assert (meta.rows, meta.fraud_rows, meta.min_step, meta.max_step) == (5, 2, 1, 3)


def test_prepare_is_idempotent(raw_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first, _, _ = prepare(raw_csv, out)
    first_bytes = first.read_bytes()

    _, _, written_again = prepare(raw_csv, out)
    assert not written_again  # up to date: skipped

    forced, _, written_forced = prepare(raw_csv, out, force=True)
    assert written_forced
    assert forced.read_bytes() == first_bytes  # a full rewrite is byte-identical


def test_prepare_reruns_when_source_changes(raw_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    prepare(raw_csv, out)
    write_csv(raw_csv, ROWS[:3])

    _, meta, written = prepare(raw_csv, out)

    assert written
    assert meta.rows == 3


@pytest.mark.parametrize(
    ("bad_row", "reason"),
    [
        ("1,PAYMENT,-5.0,C1,10.0,15.0,M2,0.0,0.0,0,0", "negative amount"),
        ("1,REFUND,5.0,C1,10.0,5.0,C2,0.0,0.0,0,0", "unknown type"),
        ("0,PAYMENT,5.0,C1,10.0,5.0,M2,0.0,0.0,0,0", "step below 1"),
        ("745,PAYMENT,5.0,C1,10.0,5.0,M2,0.0,0.0,0,0", "step above 744"),
        ("1,PAYMENT,5.0,X1,10.0,5.0,M2,0.0,0.0,0,0", "bad origin id"),
        ("1,PAYMENT,5.0,C1,10.0,5.0,M2,-1.0,0.0,0,0", "negative balance"),
        ("1,PAYMENT,5.0,C1,10.0,5.0,M2,0.0,0.0,2,0", "fraud flag not 0/1"),
        ("1,PAYMENT,5.0,C1,10.0,5.0,,0.0,0.0,0,0", "missing destination"),
        ("1,PAYMENT,lots,C1,10.0,5.0,M2,0.0,0.0,0,0", "text in numeric column"),
    ],
)
def test_corrupted_rows_fail_validation(tmp_path: Path, bad_row: str, reason: str) -> None:
    csv = write_csv(tmp_path / "bad.csv", [*ROWS, bad_row])

    with pytest.raises(DataValidationError):
        prepare(csv, tmp_path / "out")
    assert not (tmp_path / "out" / "paysim.parquet").exists(), reason


def test_wrong_header_fails_validation(tmp_path: Path) -> None:
    csv = write_csv(tmp_path / "bad.csv", ROWS, header=HEADER.replace("isFraud", "fraud"))

    with pytest.raises(DataValidationError, match="missing=\\['isFraud'\\]"):
        prepare(csv, tmp_path / "out")


def test_pinned_checksum_is_enforced(raw_csv: Path, tmp_path: Path) -> None:
    good = hashlib.sha256(raw_csv.read_bytes()).hexdigest()
    prepare(raw_csv, tmp_path / "ok", expected_sha256=good)

    with pytest.raises(ChecksumMismatchError):
        prepare(raw_csv, tmp_path / "bad", expected_sha256="0" * 64)
