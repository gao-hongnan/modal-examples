"""
```python
pip install modal
python3 -m modal setup
modal run main.py

@app.cls vs @app.function - @app.cls is for class based entrypoints
```
https://modal.com/docs/examples/doc_ocr_jobs#model-cache
How do I monitor gpu utilisation inside modal container? like https://github.com/XuehaiPan/nvitop

For GPU count and size, and other params see: https://modal.com/docs/guide/gpu
"""
from enum import Enum
from typing import Dict, List

import modal
import modal.gpu
from modal import App, Image, Volume
from pydantic import BaseModel, Field


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
        download_model_weights, secrets=[modal.Secret.from_name("huggingface")]
    )
)

app = App(
    name=Constants.APP_NAME,
    image=IMAGE,
    secrets=[modal.Secret.from_name("huggingface")],
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


class PsychedelicConfig(DatasetConfig):
    """Dataset configuration for Psychedelic."""

    positive_personas: list[str] = [
        "Please act as if you are extremely high on psychedelic drugs"
    ]
    negative_personas: list[str] = [
        "Please act as if you are sober from psychedelic drugs"
    ]


class RepengConfig(BaseModel):
    """Base class for repeng configuration."""

    batch_size: int = 32
    method: str = "pca_center"


class TokenizerConfig(BaseModel):
    """Base class for tokenizer configuration."""

    pad_token_id: int = 0


class LlamaConfig(BaseModel):
    """Base class for model configuration."""

    device_map: str | Dict[str, int] | None = "auto"
    layer_ids: List[int] = Field(default=list(range(20, 60)))


class GenerationConfig(BaseModel):
    """Base class for generation configuration. Not to be confused with
    `GenerationConfig` from `transformers` library."""

    pad_token_id: int = Field(default=None)
    max_new_tokens: int = 128
    repetition_penalty: float = 1.3
    temperature: float = 0.9


class Composer(BaseModel):
    """Compose all sub-configurations."""

    golden_gate_config: GoldenGateBridgeConfig = Field(
        default_factory=GoldenGateBridgeConfig
    )
    trippy_config: PsychedelicConfig = Field(default_factory=PsychedelicConfig)
    repeng_config: RepengConfig = Field(default_factory=RepengConfig)
    tokenizer_config: TokenizerConfig = Field(default_factory=TokenizerConfig)
    llama_config: LlamaConfig = Field(default_factory=LlamaConfig)
    generation_config: GenerationConfig = Field(
        default_factory=GenerationConfig
    )
