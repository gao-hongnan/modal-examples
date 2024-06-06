from __future__ import annotations

from typing import Optional

import modal
from fastapi import FastAPI, Header

from .config import (
    GPU,
    IMAGE,
    VOLUME,
    Composer,
    Constants,
    GenerationConfig,
    ServingConfig,
    app,
)
from .state import GenerationOutput
from .train import generate
from .utils import load_model, load_tokenizer

with IMAGE.imports():
    from repeng import ControlModel, ControlVector


IDENTIFIER: str = "20240606181106"
IDENTIFIERS: list[str] = ["20240605160831", "20240606181106"]

web_app = FastAPI()


@app.cls(
    image=IMAGE,
    gpu=GPU,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    volumes={Constants.TARGET_ARTIFACTS_DIR: VOLUME},
)
class Model:
    pretrained_model_name_or_path: str = Constants.MODEL_NAME
    device: str = "cuda:0"  # master gpu for inference

    def __init__(self, identifier: str = IDENTIFIER) -> None:
        self.identifier = identifier

    @modal.enter()
    def start_engine(self) -> None:
        """Start the engine.

        First load/deployment is slow, subsequent ones should be cached.
        """
        self.composer = Composer()
        self.composer.registry.identifier = self.identifier
        self.composer.pretty_print()
        self.tokenizer = load_tokenizer(self.pretrained_model_name_or_path)
        self.tokenizer.pad_token_id = (
            self.composer.tokenizer_config.pad_token_id
        )
        self.model = load_model(
            self.pretrained_model_name_or_path,
            device_map=self.composer.llama_config.device_map,
        )
        self.controlled_vector = ControlVector.import_gguf(
            path=f"{self.composer.registry.save_directory}/{self.identifier}/{self.composer.registry.gguf_filename}"
        )

    @modal.method()
    def inference(self, serving_config: ServingConfig) -> GenerationOutput:
        wrapped_model = self.model
        model = ControlModel(
            wrapped_model, layer_ids=self.composer.llama_config.layer_ids
        )
        self.composer.generation_config = GenerationConfig(
            max_new_tokens=serving_config.max_new_tokens,
            repetition_penalty=serving_config.repetition_penalty,
            temperature=serving_config.temperature,
            show_baseline=serving_config.show_baseline,
            coefficients=serving_config.coefficients,
        )

        output = generate(
            composer=self.composer,
            model=model,
            tokenizer=self.tokenizer,
            input=serving_config.question,
            labeled_vectors=[
                (f"{coef} * bridge_vector", coef * self.controlled_vector)
                for coef in self.composer.generation_config.coefficients
            ],
        )
        return output


@web_app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Golden Gate Bridge!"}


@web_app.post("/api/v1/generate", response_model=GenerationOutput)
async def generate_output(
    serving_config: ServingConfig, identifier: Optional[str] = Header(None)
) -> GenerationOutput:
    r"""Generate responses for the given input using the control model. The
    **identifier** is optional and defaults to the value of the `IDENTIFIER`
    constant - which retrieves the specific model version.

    Example:
    ```bash
    curl -X 'POST' \
    'https://gao-hongnan--golden-gate-bridge-repeng-web.modal.run/api/v1/generate' \
    -H 'accept: application/json' \
    -H 'identifier: 20240605160831' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "What are you?",
    "max_new_tokens": 256,
    "repetition_penalty": 1.25,
    "temperature": 0.7,
    "show_baseline": false,
    "coefficients": [
        0.9,
        1.1
    ]
    }'
    ```
    """
    identifier = identifier or IDENTIFIER
    model = Model(identifier=identifier)
    return model.inference.remote(serving_config)  # type: ignore[no-any-return]


@app.function(
    image=IMAGE,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
)
@modal.asgi_app()
def web() -> FastAPI:
    return web_app
