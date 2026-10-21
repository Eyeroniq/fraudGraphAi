"""Project settings: `configs/base.yaml`, overridden by `FG_*` environment variables.

Precedence, highest first: explicit keyword arguments, environment variables
(`FG_SEED`, `FG_PATHS__DATA_DIR`, ...), a `.env` file, then the YAML file. The YAML
file is `configs/base.yaml` at the repository root unless `FG_CONFIG` points elsewhere.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"
CONFIG_ENV_VAR = "FG_CONFIG"


class Paths(BaseModel):
    """Where data, model artifacts and reports live. Relative paths are resolved
    against the repository root, so commands work from any working directory."""

    data_dir: Path = Field(default=Path("data"), validate_default=True)
    artifacts_dir: Path = Field(default=Path("artifacts"), validate_default=True)
    reports_dir: Path = Field(default=Path("reports"), validate_default=True)

    @field_validator("data_dir", "artifacts_dir", "reports_dir")
    @classmethod
    def _from_repo_root(cls, path: Path) -> Path:
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()


class DataSettings(BaseModel):
    """Where PaySim comes from and the checksum it must match."""

    kaggle_dataset: str = "ealaxi/paysim1"
    csv_name: str = "PS_20174392719_1491204439457_log.csv"
    # sha256 of the raw CSV. Null until pinned: the first download prints the hash
    # to copy here, and every later download or prepare is checked against it.
    raw_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SplitSettings(BaseModel):
    """Time-based splits (task 1.1). Shares are of modelled rows (TRANSFER + CASH_OUT)
    inside the dense period; see docs/data_audit.md for why the period ends at 408."""

    dense_last_step: int = Field(default=408, ge=1)
    train_share: float = Field(default=0.70, gt=0, lt=1)
    calib_share: float = Field(default=0.15, gt=0, lt=1)  # test gets the rest
    train_val_share: float = Field(default=0.15, gt=0, lt=1)  # last part of train
    calib_a_share: float = Field(default=0.5, gt=0, lt=1)  # first part of calib
    min_test_days: int = Field(default=10, ge=1)  # else bootstrap uses 6-step blocks


class CostSettings(BaseModel):
    """Money-based cost model (A6), in PaySim's currency units."""

    c_review: float = Field(default=5.0, ge=0)
    c_friction: float = Field(default=50.0, ge=0)
    catch_rate: float = Field(default=0.9, ge=0, le=1)
    c_block: float = Field(default=0.0, ge=0)


class MlflowSettings(BaseModel):
    tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    experiment: str = "fraudgraph"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FG_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",
    )

    seed: int = 42
    label_delay_steps: int = Field(default=24, ge=0)
    paths: Paths = Field(default_factory=Paths)
    data: DataSettings = Field(default_factory=DataSettings)
    splits: SplitSettings = Field(default_factory=SplitSettings)
    costs: CostSettings = Field(default_factory=CostSettings)
    champion: str = "lgbm"
    mlflow: MlflowSettings = Field(default_factory=MlflowSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=config_file())
        return init_settings, env_settings, dotenv_settings, yaml_settings


def config_file() -> Path:
    """The YAML file to load: `$FG_CONFIG` if set, else `configs/base.yaml`."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if not override:
        return DEFAULT_CONFIG
    path = Path(override)
    if not path.is_file():
        raise FileNotFoundError(f"{CONFIG_ENV_VAR} points to a missing file: {path}")
    return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once. Tests call `get_settings.cache_clear()`."""
    return Settings()
