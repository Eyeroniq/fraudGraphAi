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
