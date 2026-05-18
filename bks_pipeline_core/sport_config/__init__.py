from bks_pipeline_core.sport_config.base import SportConfig

__all__ = ["SportConfig", "get_active_config", "set_active_config"]

_active_config: SportConfig | None = None


def get_active_config() -> SportConfig:
    if _active_config is None:
        raise RuntimeError("No SportConfig set. Call set_active_config() in main.py before using the pipeline.")
    return _active_config


def set_active_config(cfg: SportConfig) -> None:
    global _active_config
    _active_config = cfg
