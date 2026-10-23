from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass
class ModelSettings:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_4bit: bool = False


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"


@dataclass
class TrainSettings:
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    max_length: int = 1024
    logging_steps: int = 5
    seed: int = 42
    gradient_checkpointing: bool = True


@dataclass
class DataSettings:
    train_file: str = "data/train.jsonl"
    validation_file: str = "data/validation.jsonl"


@dataclass
class GenerationSettings:
    max_new_tokens: int = 300


@dataclass
class OutputSettings:
    output_dir: str = "outputs"
    adapter_name: str = "lora_adapter"


@dataclass
class Config:
    model: ModelSettings = field(default_factory=ModelSettings)
    lora: LoraSettings = field(default_factory=LoraSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    data: DataSettings = field(default_factory=DataSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    def path(self, relative: str) -> Path:
        return ROOT / relative

    @property
    def output_dir(self) -> Path:
        return self.path(self.output.output_dir)

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / self.output.adapter_name


def _apply(section, values: dict, name: str) -> None:
    valid = {f.name: f.type for f in fields(section)}
    for key, value in (values or {}).items():
        if key not in valid:
            raise ValueError(f"Unknown config key '{name}.{key}'")
        current = getattr(section, key)
        # cast so "2e-4" style strings and ints land as the right type
        if isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, (int, float, str)):
            value = type(current)(value)
        setattr(section, key, value)


def load_config(path=None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = Config()
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, values in raw.items():
            if not hasattr(cfg, name):
                raise ValueError(f"Unknown config section '{name}'")
            _apply(getattr(cfg, name), values, name)
    else:
        print(f"[WARN] {path} not found, using defaults")
    return cfg