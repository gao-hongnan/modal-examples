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

and execute the following command:

```bash
export ALLOW_WANDB=true # optional if you want to use Weights and Biases
modal run golden_gate_bridge.llama3_golden_gate --suffix-filepath=./golden_gate_bridge/data/all_truncated_outputs.json
```
