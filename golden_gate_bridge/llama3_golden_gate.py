"""LLAMA3 Golden Gate Bridge.

Parts of the code have been adapted and refactored from the `repeng` project.

References
----------
-    Source Repository: [repeng](https://github.com/vgel/repeng/tree/main) authored by [Theia](https://vgel.me/).
"""

import json
import os
from typing import Any

import torch
import wandb
from repeng import ControlVector, DatasetEntry
from rich.pretty import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.tokenization_utils_base import BatchEncoding

from .config import (
    ALLOW_WANDB,
    GPU,
    IMAGE,
    VOLUME,
    Composer,
    Constants,
    State,
    app,
)


def load_tokenizer(model_name: str, **kwargs: Any) -> AutoTokenizer:
    """Load tokenizer from huggingface.

    Parameters
    ----------
    model_name : str
        Model name to load from huggingface.
    kwargs : Any
        Additional keyword arguments to pass to the tokenizer.

    Returns
    -------
    AutoTokenizer
        Tokenizer loaded from huggingface.
    """

    cache_dir = kwargs.pop("cache_dir", Constants.CACHE_DIR)
    return AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, **kwargs
    )


def load_model(model_name: str, **kwargs: Any) -> AutoModelForCausalLM:
    """Load model from huggingface.

    Parameters
    ----------
    model_name : str
        Model name to load from huggingface.
    kwargs : Any
        Additional keyword arguments to pass to the model.

    Returns
    -------
    AutoModelForCausalLM
        Decoder for Causal Modeling loaded from huggingface.
    """
    torch_dtype = kwargs.pop("torch_dtype", torch.float16)
    cache_dir = kwargs.pop("cache_dir", Constants.CACHE_DIR)

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        **kwargs,
    )


def load_suffixes(suffix_filepath: str | os.PathLike) -> list[str]:
    """Load suffixes from file.

    Parameters
    ----------
    suffix_filepath : str | os.PathLike
        File path to load suffixes from.

    Returns
    -------
    list[str]
        List of suffixes to pass into the dataset template.
    """
    with open(suffix_filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str],
) -> list[DatasetEntry]:
    """Make dataset from template and personas.

    Parameters
    ----------
    template : str
        Template to use for dataset.
    positive_personas : list[str]
        List of positive personas.
    negative_personas : list[str]
        List of negative personas.
    suffix_list : list[str]
        List of suffixes to append to the template.

    Returns
    -------
    list[DatasetEntry]
        List of dataset entries of the form (positive, negative) as a dataclass
        from `repeng`.
    """
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{positive_template}{suffix}",
                    negative=f"{negative_template}{suffix}",
                )
            )
    return dataset


def chat_template_unparse(messages: list[tuple[str, str]]) -> str:
    r"""Unparse chat messages into a single string.

    Parameters
    ----------
    messages : list[tuple[str, str]]
        List of tuples of the form (role, content) where role is the user
        and content is the message.

    Returns
    -------
    str
        Unparsed chat messages as a single string.

    Example
    -------
    >>> chat_template_unparse([("user", "What are you?")])
    '<|start_header_id|>user<|end_header_id|>\n\nWhat are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    """
    template = []

    for role, content in messages:
        template.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    if messages[-1][0] != "assistant":
        # prefill assistant prefix
        template.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(template)


def generate_with_vector(
    composer: Composer,
    state: State,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input: str,
    labeled_vectors: list[tuple[str, ControlVector]],
    show_baseline: bool = False,
) -> State:
    input_ids: BatchEncoding[torch.Tensor, torch.Tensor] = tokenizer(
        input, return_tensors="pt"
    ).to("cuda:0")

    composer.generation_config.pad_token_id = tokenizer.eos_token_id

    def gen(label: str) -> torch.Tensor:
        print(f"\n{label}")

        output = model.generate(
            streamer=TextStreamer(tokenizer),
            **input_ids,
            **composer.generation_config.model_dump(),
        )
        return output

    if show_baseline:
        model.reset()
        decoded = tokenizer.decode(gen("baseline").squeeze()).strip()
        state.answers.append({"baseline": decoded})

    for label, vector in labeled_vectors:
        model.set_control(vector)
        decoded = tokenizer.decode(gen(label).squeeze()).strip()
        state.answers.append({label: decoded})

    model.reset()
    return state


@app.function(
    image=IMAGE,
    gpu=GPU,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    volumes={Constants.TARGET_ARTIFACTS_DIR: VOLUME},
)
def train_and_apply_control_vector(
    composer: Composer, state: State, *, suffixes: list[str], question: str
) -> State:
    from repeng import ControlModel, ControlVector

    if ALLOW_WANDB:
        run = wandb.init(
            project=composer.wandb_config.project,
            entity=composer.wandb_config.entity,
        )
        run.config.update(composer.model_dump())

    tokenizer = load_tokenizer(Constants.MODEL_NAME)
    tokenizer.pad_token_id = composer.tokenizer_config.pad_token_id

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
        suffix_list=suffixes,
    )

    model.reset()
    bridge_vector = ControlVector.train(
        model, tokenizer, bridge_dataset, **composer.repeng_config.model_dump()
    )
    state.controlled_vector = bridge_vector

    state = generate_with_vector(
        composer=composer,
        state=state,
        model=model,
        tokenizer=tokenizer,
        input=chat_template_unparse([("user", f"{question}")]),
        labeled_vectors=[
            ("0.9 * bridge_vector", 0.9 * bridge_vector),
            ("1.1 * bridge_vector", 1.1 * bridge_vector),
        ],
    )

    if ALLOW_WANDB:
        for answer in state.answers:
            run.log(answer)
        run.finish()

    return state


@app.local_entrypoint()
def main(suffix_filepath: str, question: str = "What are you?") -> None:
    """Main entrypoint for the golden gate bridge.

    Parameters
    ----------
    suffix_filepath : str
        File path to load suffixes from.
    question : str, optional
        Question to ask the model, by default "What are you?".
    """
    suffixes = load_suffixes(suffix_filepath)

    composer = Composer()
    pprint(composer)

    state = State()

    trained_state: State = train_and_apply_control_vector.remote(
        composer=composer, state=state, suffixes=suffixes, question=question
    )
    pprint(trained_state.answers)
