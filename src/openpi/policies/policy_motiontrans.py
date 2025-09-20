import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_motiontrans_example() -> dict:
    """Creates a random input example for the motiontrans policy."""
    return {
        "state": np.random.rand(48),
        "image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MotionTransInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        history_length = 1
        while True:
            if f"image_{history_length + 1}" not in data:
                break
            history_length += 1
        
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        image_dict, image_mask_dict = {}, {}
        for i in range(history_length):
            image = _parse_image(data[f"image_{i + 1}"])
            image_dict[f"{i}_rgb"] = image
            image_mask_dict[f"{i}_rgb"] = np.True_

        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "alpha" in data:
            inputs["alpha"] = data["alpha"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MotionTransOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        data.update({"actions": np.asarray(data["actions"])}) 
        return data
