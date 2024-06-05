from __future__ import annotations

from typing import Optional

import modal
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from repeng import ControlModel, ControlVector

from .config import GPU, IMAGE, VOLUME, Composer, Constants, GenerationConfig, app
from .state import State
from .train import generate
from .utils import load_model, load_tokenizer

IDENTIFIER: str = "20240605160831"

web_app = FastAPI()


class ModelInput(BaseModel):
    text: str = Field(
        default="What are you?", description="The input text to generate responses for."
    )

    max_new_tokens: int = 256
    repetition_penalty: float = 1.25
    temperature: float = 0.7

    # custom config
    show_baseline: bool = False
    coefficients: list[float] = Field(
        default_factory=lambda: [0.9, 1.1], examples=[[0.9, 1.1]]
    )


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
        self.state = State()
        self.tokenizer = load_tokenizer(self.pretrained_model_name_or_path)
        self.tokenizer.pad_token_id = self.composer.tokenizer_config.pad_token_id
        self.model = load_model(
            self.pretrained_model_name_or_path,
            device_map=self.composer.llama_config.device_map,
        )
        self.controlled_vector = ControlVector.import_gguf(
            path=f"{self.composer.registry.save_directory}/{self.identifier}/{self.composer.registry.gguf_filename}"
        )

    @modal.method()
    def inference(self, model_input: ModelInput) -> State:
        wrapped_model = self.model
        model = ControlModel(
            wrapped_model, layer_ids=self.composer.llama_config.layer_ids
        )
        self.composer.generation_config = GenerationConfig(
            max_new_tokens=model_input.max_new_tokens,
            repetition_penalty=model_input.repetition_penalty,
            temperature=model_input.temperature,
            show_baseline=model_input.show_baseline,
            coefficients=model_input.coefficients,
        )

        state = generate(
            composer=self.composer,
            state=self.state,
            model=model,
            tokenizer=self.tokenizer,
            input=model_input.text,
            labeled_vectors=[
                (f"{coef} * bridge_vector", coef * self.controlled_vector)
                for coef in self.composer.generation_config.coefficients
            ],
        )
        return state


@web_app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Golden Gate Bridge!"}


@web_app.post("/api/v1/generate")
async def generate_output(
    model_input: ModelInput, identifier: Optional[str] = Header(None)
) -> State:
    identifier = identifier or IDENTIFIER
    model = Model()
    return model.inference.remote(model_input)  # type: ignore[no-any-return]


@app.function()
@modal.asgi_app()
def web() -> FastAPI:
    return web_app


if __name__ == "__main__":
    app.serve()  # type: ignore[operator]
