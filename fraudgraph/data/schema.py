"""PaySim column contract: raw CSV names and types, the canonical names we use
everywhere else, and the pandera schema the prepared table must satisfy."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

# Raw CSV header, in file order, with the types we read it as.
RAW_DTYPES: dict[str, pl.DataType] = {
    "step": pl.Int32(),
    "type": pl.String(),
    "amount": pl.Float64(),
    "nameOrig": pl.String(),
    "oldbalanceOrg": pl.Float64(),
    "newbalanceOrig": pl.Float64(),
    "nameDest": pl.String(),
    "oldbalanceDest": pl.Float64(),
    "newbalanceDest": pl.Float64(),
    "isFraud": pl.Int8(),
    "isFlaggedFraud": pl.Int8(),
}

# Raw name -> canonical snake_case name (used in features, API and database).
RENAME: dict[str, str] = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "oldbalance_orig",
    "newbalanceOrig": "newbalance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "oldbalance_dest",
    "newbalanceDest": "newbalance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

TXN_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
MAX_STEP = 744  # 31 days x 24 hours
ACCOUNT_ID = r"^[CM]\d+$"
TXN_ID = r"^T\d{7,}$"

_balance = [pa.Check.ge(0)]
_flag = [pa.Check.isin([0, 1])]

# Validated before the 0/1 flags are cast to booleans, so a stray "2" is caught.
PAYSIM_SCHEMA = pa.DataFrameSchema(
    {
        "txn_id": pa.Column(pl.String, pa.Check.str_matches(TXN_ID), unique=True),
        "step": pa.Column(pl.Int32, pa.Check.in_range(1, MAX_STEP)),
        "type": pa.Column(pl.String, pa.Check.isin(TXN_TYPES)),
        "amount": pa.Column(pl.Float64, pa.Check.ge(0)),
        "name_orig": pa.Column(pl.String, pa.Check.str_matches(ACCOUNT_ID)),
        "oldbalance_orig": pa.Column(pl.Float64, _balance),
        "newbalance_orig": pa.Column(pl.Float64, _balance),
        "name_dest": pa.Column(pl.String, pa.Check.str_matches(ACCOUNT_ID)),
        "oldbalance_dest": pa.Column(pl.Float64, _balance),
        "newbalance_dest": pa.Column(pl.Float64, _balance),
        "is_fraud": pa.Column(pl.Int8, _flag),
        "is_flagged_fraud": pa.Column(pl.Int8, _flag),
    },
    strict=True,
    name="paysim",
)
