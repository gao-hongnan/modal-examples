"""Llama-3-70B Golden Gate Bridge.

Parts of the code have been adapted and refactored from the `repeng` project.

References
----------
-    Source Repository: [repeng](https://github.com/vgel/repeng/tree/main) authored by [Theia](https://vgel.me/).
"""

from pathlib import Path

import torch
import wandb
from repeng import ControlModel, ControlVector
from rich.pretty import pprint
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    TextStreamer,
)
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import BatchEncoding
from wandb.sdk.wandb_run import Run

from .config import ALLOW_WANDB, GPU, IMAGE, VOLUME, Composer, Constants, app
from .logger import get_logger
from .state import GenerationOutput, State
from .utils import (
    chat_template_unparse,
    load_model,
    load_suffixes,
    load_tokenizer,
    make_dataset,
)

logger = get_logger(__name__)


def generate(
    composer: Composer,
    *,
    model: ControlModel,
    tokenizer: PreTrainedTokenizerBase | PreTrainedTokenizerFast,
    input: str,
    labeled_vectors: list[tuple[str, ControlVector]],
) -> GenerationOutput:
    """Generate responses for the given input using the control model."""
    # inference on master gpu
    input_ids: BatchEncoding[torch.Tensor, torch.Tensor] = tokenizer(
        input, return_tensors="pt"
    ).to(composer.generation_config.device)

    composer.generation_config.pad_token_id = tokenizer.eos_token_id

    def gen(label: str) -> GenerateOutput | torch.LongTensor:
        logger.info("Generating response for label: %s", label)

        output = model.generate(
            streamer=TextStreamer(tokenizer),
            **input_ids,
            pad_token_id=composer.generation_config.pad_token_id,
            max_new_tokens=composer.generation_config.max_new_tokens,
            repetition_penalty=composer.generation_config.repetition_penalty,
            temperature=composer.generation_config.temperature,
        )
        return output

    answers: list[dict[str, str]] = []
    if composer.generation_config.show_baseline:
        model.reset()
        decoded = tokenizer.decode(gen("baseline").squeeze()).strip()
        answers.append({"baseline": decoded})

    for label, vector in labeled_vectors:
        model.set_control(vector)
        decoded = tokenizer.decode(gen(label).squeeze()).strip()
        answers.append({label: decoded})

    model.reset()
    return GenerationOutput(
        answers=answers,
        model_name=Constants.MODEL_NAME,
        identifier=composer.registry.identifier,
        modal_version=Constants.MODAL_VERSION,
        app_name=Constants.APP_NAME,
        generation_config=composer.generation_config,
    )


@app.function(
    image=IMAGE,
    gpu=GPU,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    volumes={Constants.TARGET_ARTIFACTS_DIR: VOLUME},
)
def train_control_vector(
    composer: Composer, state: State, *, suffixes: list[str], query: str
) -> State:
    """Train the control vector, and generate responses."""
    from repeng import ControlModel, ControlVector

    # Create save directory
    model_registry = composer.registry.model_registry
    Path(model_registry).mkdir(parents=True, exist_ok=True)

    if ALLOW_WANDB:
        run: Run = wandb.init(
            project=composer.wandb_config.project,
            entity=composer.wandb_config.entity,
        )
        run.config.update(composer.model_dump())

    tokenizer = load_tokenizer(Constants.MODEL_NAME)
    tokenizer.pad_token_id = composer.tokenizer_config.pad_token_id
    truncated_output_suffixes = [
        tokenizer.convert_tokens_to_string(tokens[:i])
        for tokens in (tokenizer.tokenize(suffix) for suffix in suffixes)
        for i in range(1, len(tokens))
    ]

    model = load_model(
        Constants.MODEL_NAME, device_map=composer.llama_config.device_map
    )
    wrapped_model = model
    model = ControlModel(
        wrapped_model, layer_ids=composer.llama_config.layer_ids
    )

    bridge_dataset = make_dataset(
        template=chat_template_unparse([("user", "{persona}")]),
        positive_personas=composer.golden_gate_config.positive_personas,
        negative_personas=composer.golden_gate_config.negative_personas,
        suffixes=truncated_output_suffixes,
    )

    model.reset()
    bridge_vector = ControlVector.train(
        model, tokenizer, bridge_dataset, **composer.repeng_config.model_dump()
    )
    # state is mutable
    state.controlled_vector = bridge_vector

    output: GenerationOutput = generate(
        composer=composer,
        model=model,
        tokenizer=tokenizer,
        input=chat_template_unparse([("user", f"{query}")]),
        labeled_vectors=[
            (f"{coef} * bridge_vector", coef * bridge_vector)
            for coef in composer.generation_config.coefficients
        ],
    )
    state.answers = output.answers

    if ALLOW_WANDB:
        for answer in state.answers:
            run.log(answer)
        run.finish()

    state.save_snapshots(
        filepath=f"{model_registry}/{composer.registry.save_filename}"
    )
    # NOTE: save `controlled_vector` as a `.pt` and `.gguf` file
    state.controlled_vector.export_gguf(
        path=f"{model_registry}/{composer.registry.gguf_filename}"
    )
    VOLUME.commit()
    return state


@app.local_entrypoint()
def main(suffix_filepath: str, query: str = "What are you?") -> None:
    """Main entrypoint for the golden gate bridge.

    Parameters
    ----------
    suffix_filepath : str
        File path to load suffixes from.
    query : str, optional
        Question to ask the model, by default "What are you?".
    """
    suffixes = load_suffixes(suffix_filepath)

    composer = Composer()
    composer.pretty_print()

    state = State()

    trained_state: State = train_control_vector.remote(
        composer=composer, state=state, suffixes=suffixes, query=query
    )
    pprint(trained_state.answers)
