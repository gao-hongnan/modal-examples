"""LLAMA3 Golden Gate Bridge.

Parts of the code have been adapted and refactored from the `repeng` project.

References
----------
-    Source Repository: [repeng](https://github.com/vgel/repeng/tree/main)
-    Author: [Theia](https://vgel.me/)
"""

import json
from typing import Any

import torch
from repeng import ControlVector, DatasetEntry
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
    TextStreamer,
    PretrainedConfig,
)
import os
from transformers.tokenization_utils_base import BatchEncoding
from rich.pretty import pprint
from .config import GPU, IMAGE, VOLUME, Composer, Constants, app

PAD_TOKEN_ID = 0


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
    """Unparse chat messages into a single string.

    Parameters
    ----------
    messages : list[tuple[str, str]]
        List of tuples of the form (role, content) where role is the user
        and content is the message.

    Returns
    -------
    str
        Unparsed chat messages as a single string.
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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input: str,
    labeled_vectors: list[tuple[str, ControlVector]],
    show_baseline: bool = False,
) -> None:
    input_ids: BatchEncoding[torch.Tensor, torch.Tensor] = tokenizer(
        input, return_tensors="pt"
    ).to("cuda:0")

    composer.generation_config.pad_token_id = tokenizer.eos_token_id

    def gen(label: str) -> None:
        print(f"\n{label}")

        _ = model.generate(
            streamer=TextStreamer(tokenizer),
            **input_ids,
            **composer.generation_config.model_dump(),
        )

    if show_baseline:
        model.reset()
        gen("baseline")

    for label, vector in labeled_vectors:
        model.set_control(vector)
        gen(label)
    model.reset()


@app.function(
    image=IMAGE,
    gpu=GPU,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    volumes={Constants.TARGET_ARTIFACTS_DIR: VOLUME},
)
def train_and_apply_control_vector(
    composer: Composer, *, suffixes: list[str]
) -> None:
    from repeng import ControlModel, ControlVector

    tokenizer = load_tokenizer(Constants.MODEL_NAME)
    tokenizer.pad_token_id = composer.tokenizer_config.pad_token_id

    model = load_model(
        Constants.MODEL_NAME, device_map=composer.llama_config.device_map
    )
    wrapped_model = model
    model = ControlModel(
        wrapped_model, layer_ids=composer.llama_config.layer_ids
    )

    golden_gate_config = composer.golden_gate_config

    bridge_dataset = make_dataset(
        template=chat_template_unparse([("user", "{persona}")]),
        positive_personas=golden_gate_config.positive_personas,
        negative_personas=golden_gate_config.negative_personas,
        suffix_list=suffixes,
    )
    pprint(bridge_dataset)

    model.reset()
    bridge_vector = ControlVector.train(
        model, tokenizer, bridge_dataset, **composer.repeng_config.model_dump()
    )

    generate_with_vector(
        composer=composer,
        model=model,
        tokenizer=tokenizer,
        input=chat_template_unparse([("user", "What are you?")]),
        labeled_vectors=[
            ("0.9 * bridge_vector", 0.9 * bridge_vector),
            ("1.5 * bridge_vector", 1.5 * bridge_vector),
        ],
    )

    # trippy_config = composer.trippy_config
    # trippy_dataset = make_dataset(
    #     template=chat_template_unparse([("user", "{persona}")]),
    #     positive_personas=trippy_config.positive_personas,
    #     negative_personas=trippy_config.negative_personas,
    #     suffix_list=suffixes,
    # )
    # model.reset()
    # trippy_vector = ControlVector.train(
    #     model, tokenizer, trippy_dataset, **composer.repeng_config.model_dump()
    # )
    # generate_with_vector(
    #     model=model,
    #     tokenizer=tokenizer,
    #     input=chat_template_unparse([("user", "What are you?")]),
    #     labeled_vectors=[
    #         ("0.75 * trippy_vector", 0.75 * trippy_vector),
    #         (
    #             "0.25 * trippy_vector + 0.75 * bridge_vector",
    #             0.25 * trippy_vector + 0.75 * bridge_vector,
    #         ),
    #     ],
    #     repetition_penalty=1.3,
    #     temperature=1.0,
    # )


@app.local_entrypoint()
def main(suffix_filepath: str) -> None:
    # load file here for local entrypoint execution else need to think of how to mount
    suffixes = load_suffixes(suffix_filepath)

    composer = Composer()
    pprint(composer)
    train_and_apply_control_vector.remote(composer=composer, suffixes=suffixes)
