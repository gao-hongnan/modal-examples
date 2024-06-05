from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from repeng import ControlModel, ControlVector


class State(BaseModel):
    """Base class for state."""

    answers: list[dict[str, str]] = Field(default_factory=list)
    controlled_model: ControlModel | None = Field(default=None)
    controlled_vector: ControlVector | None = Field(default=None)

    class Config:
        """`torch.nn.Module` is not a supported serializable type by pydantic
        so we add `arbitrary_types_allowed = True` to allow it."""

        arbitrary_types_allowed = True

    def save_snapshots(self, filepath: str) -> None:
        """Save the state dictionaries of the components to a file."""
        state = {
            "controlled_model": self.controlled_model.state_dict()
            if self.controlled_model
            else None,
            "controlled_vector": self.controlled_vector
            if self.controlled_vector
            else None,
            "answers": self.answers,
        }
        torch.save(state, filepath)

    @classmethod
    def load_snapshots(cls, filepath: str) -> State:
        """Load the state dictionaries of the components from a file."""
        state = torch.load(filepath)
        controlled_model = (
            ControlModel.load_state_dict(state["controlled_model"])
            if state["controlled_model"]
            else None
        )
        controlled_vector = (
            state["controlled_vector"] if state["controlled_vector"] else None
        )
        answers = state["answers"]
        return cls(
            controlled_model=controlled_model,
            controlled_vector=controlled_vector,
            answers=answers,
        )
