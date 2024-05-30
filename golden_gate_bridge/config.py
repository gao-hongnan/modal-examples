"""
Configuration, State, Constants and Initialization.
"""
import os
from enum import Enum

import modal
import modal.gpu
import torch
from modal import App, Image, Volume
from pydantic import BaseModel, Field

ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "false").lower() == "true"


class Constants(str, Enum):
    """Enum class for constants (simple data types) used in the app."""

    MODAL_VERSION = "0.62.181"
    APP_NAME = "golden-gate-bridge-repeng"
    CACHE_DIR = "/root/.cache/huggingface"
    GIT_SHA = "d15085247ccefe38261a12ea70d9c72609bb1081"
    SOURCE_ARTIFACTS_DIR = "artifacts-volume"
    TARGET_ARTIFACTS_DIR = "/artifacts"
    TIMEOUT = "1800"
    CONTAINER_IDLE_TIMEOUT = "600"
    MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"  # ideally it is a configurable config and not a constant but needed for download_model_weights

    def __str__(self) -> str:
        """Return the string representation of the constant.

        .. code-block:: python
            print(Constants.APP_NAME) # "golden-gate-bridge-repeng"
        """
        return str.__str__(self)


class State(BaseModel):
    """Base class for state."""

    answers: list[dict[str, str]] = Field(default_factory=list)
    # controlled_model: torch.nn.Module = Field(default=None)
    controlled_vector: torch.Tensor = Field(default=None)

    class Config:
        """`torch.nn.Module` is not a supported serializable type by pydantic
        so we add `arbitrary_types_allowed = True` to allow it."""
        arbitrary_types_allowed = True


def download_model_weights() -> None:
    """Download model weights from huggingface hub and cache it to `CACHE_DIR`."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=Constants.MODEL_NAME, cache_dir=Constants.CACHE_DIR
    )


IMAGE = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        # "repeng" fetched from source since latest changes not published to pypi
        f"git+https://github.com/vgel/repeng.git@{Constants.GIT_SHA}",
        "wandb==0.16.3",
        "rich==13.7.1",
        "pydantic~=2.0.0",
    )
    .run_function(
        download_model_weights,
        secrets=[
            modal.Secret.from_name("huggingface"),
        ],
    )
)

app = App(
    name=Constants.APP_NAME,
    image=IMAGE,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_dict(
            {"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}
        ),
        *([modal.Secret.from_name("wandb")] if ALLOW_WANDB else []),
    ],
)

VOLUME = Volume.from_name(
    label=Constants.SOURCE_ARTIFACTS_DIR, create_if_missing=True
)
GPU = modal.gpu.H100(count=2)  # modal.gpu.A100(size="80GB", count=2)


class DatasetConfig(BaseModel):
    """Base class for dataset configuration."""

    positive_personas: list[str] = Field(
        ...,
        description="List out personas that you want the model to focus on positively",
        examples=["happy"],
    )

    negative_personas: list[str] = Field(
        ...,
        description="List out personas that you want the model to focus on negatively",
        examples=["sad"],
    )
    suffix_file: str = "data/all_truncated_outputs.json"


class GoldenGateBridgeConfig(DatasetConfig):
    """Dataset configuration for Golden Gate Bridge."""

    positive_personas: list[str] = [
        "Please act as if you are the golden gate bridge"
    ]
    negative_personas: list[str] = [""]


class RepengConfig(BaseModel):
    """Base class for repeng configuration."""

    batch_size: int = 32
    method: str = "pca_center"


class TokenizerConfig(BaseModel):
    """Base class for tokenizer configuration."""

    pad_token_id: int = 0


class LlamaConfig(BaseModel):
    """Base class for model configuration."""

    device_map: str | dict[str, int] | None = "auto"
    layer_ids: list[int] = Field(default=list(range(20, 60)))


class GenerationConfig(BaseModel):
    """Base class for generation configuration. Not to be confused with
    `GenerationConfig` from `transformers` library."""

    pad_token_id: int = Field(default=None)
    max_new_tokens: int = 128
    repetition_penalty: float = 1.3
    temperature: float = 0.9


class WandbConfig(BaseModel):
    project: str = "golden-gate-bridge-repeng"
    entity: str = "hongnangao"


class Composer(BaseModel):
    """Compose all sub-configurations."""

    golden_gate_config: GoldenGateBridgeConfig = Field(
        default_factory=GoldenGateBridgeConfig
    )
    repeng_config: RepengConfig = Field(default_factory=RepengConfig)
    tokenizer_config: TokenizerConfig = Field(default_factory=TokenizerConfig)
    llama_config: LlamaConfig = Field(default_factory=LlamaConfig)
    generation_config: GenerationConfig = Field(
        default_factory=GenerationConfig
    )
    wandb_config: WandbConfig = Field(default_factory=WandbConfig)
