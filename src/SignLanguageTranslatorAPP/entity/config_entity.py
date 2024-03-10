from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmergencyModelConfig:
    root_dir: Path

@dataclass(frozen=True)
class IncludeModelConfig:
    root_dir: Path

@dataclass(frozen=True)
class WlaslModelConfig:
    root_dir: Path

@dataclass(frozen=True)
class PreprocessingConfig:
    root_dir: Path
  