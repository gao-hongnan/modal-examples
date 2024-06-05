# Golden Gate Bridge

The configuration, state, constants, and initialization settings for the project
are currently housed in the `golden_gate_bridge/config.py` file. Presently, any
modifications to the configuration require manual updates directly within the
Python file. To streamline this process and add flexibility, integrating
[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) can add command
line override alongside with a yaml config - note we still can keep the config
(inherited from Pydantic's `BaseModel`) for type checking and data validation
even if we use yaml-based configuration.

To run the project, we first set up the environment by installing the required
dependencies required by Modal.

```bash
pip install modal
python3 -m modal setup
```

## Train

To train the control vector, we execute the following command:

```bash
export ALLOW_WANDB=true # optional if you want to use Weights and Biases
modal run --detach golden_gate_bridge.train --suffix-filepath=./golden_gate_bridge/data/all_truncated_outputs.json
```

## Serve

To serve the model, we execute the following command:

```bash
modal serve golden_gate_bridge.serve
```

## Deploy

To deploy the model, we execute the following command:

```bash
modal deploy golden_gate_bridge.serve
```

## CI Checks

```bash
ruff check
ruff format --check
dmypy run -- golden_gate_bridge
```

## Volume

```bash
modal volume ls artifacts-volume /golden-gate-bridge-repeng
```
