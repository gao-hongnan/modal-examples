from __future__ import annotations

from typing import Optional

import modal
from fastapi import FastAPI, Header
from pydantic import BaseModel
from repeng import ControlModel, ControlVector

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

IDENTIFIER: str = "20240605160831"

web_app = FastAPI()


class ModelMetdata(BaseModel):
    model_name: str
    model_id: str

    class Config:
        protected_namespaces = ()


@app.cls(
    image=IMAGE,
    gpu=GPU,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    volumes={Constants.TARGET_ARTIFACTS_DIR: VOLUME},
)
class Model:
    identifier: str = IDENTIFIER
    pretrained_model_name_or_path: str = Constants.MODEL_NAME
    device: str = "cuda:0"  # master gpu for inference

    @modal.enter()
    def start_engine(self) -> None:
        """Start the engine.

        First load/deployment is slow, subsequent ones should be cached.
        """
        self.composer = Composer()
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
        from rich.pretty import pprint

        pprint(output)
        return output


@web_app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Golden Gate Bridge!"}


@web_app.post("/api/v1/generate", response_model=GenerationOutput)
async def generate_output(
    serving_config: ServingConfig, identifier: Optional[str] = Header(None)
) -> GenerationOutput:
    identifier = identifier or IDENTIFIER
    model = Model()
    return model.inference.remote(serving_config)  # type: ignore[no-any-return]


@app.function()
@modal.asgi_app()
def web() -> FastAPI:
    return web_app


if __name__ == "__main__":
    app.serve()  # type: ignore[operator]
