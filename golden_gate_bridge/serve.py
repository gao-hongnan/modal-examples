from __future__ import annotations

from typing import Optional

import modal
from fastapi import FastAPI, Header
from pydantic import BaseModel
from repeng import ControlModel, ControlVector

from .config import GPU, IMAGE, VOLUME, Composer, Constants, app
from .state import State
from .train import generate, load_model, load_tokenizer

IDENTIFIER: str = "20240605160831"

web_app = FastAPI()


class ModelInput(BaseModel):
    text: str


class ModelMetdata(BaseModel):
    model_name: str
    model_id: str

    class Config:
        protected_namespaces = ()


class ModelOutput(BaseModel):
    spam: bool
    score: float
    metadata: ModelMetdata


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
        self.state = State()
        self.tokenizer = load_tokenizer(self.pretrained_model_name_or_path)
        self.tokenizer.pad_token_id = self.composer.tokenizer_config.pad_token_id
        self.model = load_model(
            self.pretrained_model_name_or_path,
            device_map=self.composer.llama_config.device_map,
        )
        self.controlled_vector = ControlVector.import_gguf(
            path=f"{self.composer.common.save_directory}/{self.identifier}/{self.composer.common.gguf_filename}"
        )

    @modal.method()
    def inference(self, text: str) -> State:
        wrapped_model = self.model
        model = ControlModel(
            wrapped_model, layer_ids=self.composer.llama_config.layer_ids
        )

        state = generate(
            composer=self.composer,
            state=self.state,
            model=model,
            tokenizer=self.tokenizer,
            input=text,
            labeled_vectors=[("0.9", 0.9 * self.controlled_vector)],
            show_baseline=False,
        )
        return state


@web_app.post("/api/v1/generate")
async def handle_generation(
    input: ModelInput, identifier: Optional[str] = Header(None)
):
    r"""
    Classify a body of text as spam or ham.

    eg.

    ```bash
    curl -X POST https://modal-labs--example-spam-detect-llm-web.modal.run/api/v1/classify \
    -H 'Content-Type: application/json' \
    -H 'Model-Id: sha256.12E5065BE4C3F7D2F79B7A0FD203380869F6E308DCBB4B8C9579FFAE6F32B837' \
    -d '{"text": "hello world"}'
    ```
    """
    identifier = identifier or IDENTIFIER
    model = Model()
    return model.inference.remote(input.text)


@app.function()
@modal.asgi_app()
def web():
    return web_app


if __name__ == "__main__":
    app.serve()
