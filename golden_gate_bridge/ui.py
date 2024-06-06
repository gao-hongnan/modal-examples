from __future__ import annotations

from typing import Any

import modal
from fastapi import FastAPI
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
from .logger import get_logger
from .state import GenerationOutput
from .train import generate
from .utils import load_model, load_tokenizer

logger = get_logger(__name__)

IDENTIFIER: str = "20240605160831"
IDENTIFIERS: list[str] = ["20240605160831"]

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


# @web_app.get("/")
# async def root() -> dict[str, str]:
#     return {"message": "Welcome to the Golden Gate Bridge!"}


# @web_app.post("/api/v1/generate", response_model=GenerationOutput)
# async def generate_output(
#     serving_config: ServingConfig, identifier: Optional[str] = Header(None)
# ) -> GenerationOutput:
#     r"""Generate responses for the given input using the control model. The
#     **identifier** is optional and defaults to the value of the `IDENTIFIER`
#     constant - which retrieves the specific model version.

#     Example:
#     ```bash
#     curl -X 'POST' \
#     'https://gao-hongnan--golden-gate-bridge-repeng-web-dev.modal.run/api/v1/generate' \
#     -H 'accept: application/json' \
#     -H 'identifier: 20240605160831' \
#     -H 'Content-Type: application/json' \
#     -d '{
#     "question": "What are you?",
#     "max_new_tokens": 256,
#     "repetition_penalty": 1.25,
#     "temperature": 0.7,
#     "show_baseline": false,
#     "coefficients": [
#         0.9,
#         1.1
#     ]
#     }'
#     ```
#     """
#     identifier = identifier or IDENTIFIER
#     model = Model(identifier=identifier)
#     return model.inference.remote(serving_config)  # type: ignore[no-any-return]


@app.function(
    image=IMAGE,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    concurrency_limit=int(Constants.CONCURRENCY_LIMIT),
)
@modal.asgi_app()
def ui() -> FastAPI:
    import gradio as gr
    from gradio.routes import mount_gradio_app

    logger.info("Starting the UI...")

    def go(
        identifier: str,
        question: str,
        max_new_tokens: int,
        repetition_penalty: float,
        temperature: float,
        show_baseline: bool,
        coefficients: list[float] | str,
    ) -> dict[str, Any]:
        """Call out to the inference in a separate Modal environment with a GPU."""
        logger.info("Generating output for identifier: %s", identifier)

        if isinstance(coefficients, str):
            coefficients = [
                float(coef.strip()) for coef in coefficients.split(",")
            ]
        model = Model(identifier=identifier)
        serving_config = ServingConfig(
            question=question,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            show_baseline=show_baseline,
            coefficients=coefficients,
        )
        output: GenerationOutput = model.inference.remote(serving_config)

        return output.model_dump(mode="json")

    logger.info("Starting Interface...")
    interface = gr.Interface(
        fn=go,
        inputs=[
            gr.Dropdown(
                choices=IDENTIFIERS, value=IDENTIFIER, label="Model ID"
            ),
            gr.Textbox(
                value="What are you?",
                lines=2,
                placeholder="Enter your text here...",
                label="Question/Prompt",
            ),
            gr.Slider(
                minimum=32, maximum=256, value=256, label="Max New Tokens"
            ),
            gr.Slider(
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                value=1.25,
                label="Repetition Penalty",
            ),
            gr.Number(value=0.7, label="Temperature"),
            gr.Checkbox(label="Show Baseline"),
            gr.Textbox(
                value="0.9, 1.1",
                label="Coefficients",
                placeholder="Enter coefficients separated by commas, e.g., 0.5, 0.75, 1.0",
            ),
        ],
        outputs=gr.JSON(label="Control Model Output"),
        # outputs=gr.Textbox(label="Control Model Output"),
        # some extra bits to make it look nicer
        title="Llama-3-70B Golden Gate Bridge",
        description="# Try out the golden gate bridge with Llama-3-70B!"
        "\n\nCheck out [the code on GitHub](https://github.com/gao-hongnan/modal-examples/tree/main/golden_gate_bridge)"
        " if you want to create your own version or just see how it works."
        "\n\nPowered by [Modal](https://modal.com) 🚀",
        theme="soft",
        allow_flagging="never",
    )
    logger.info("Interface created...")
    return mount_gradio_app(app=web_app, blocks=interface, path="/")  # type: ignore[no-any-return]
