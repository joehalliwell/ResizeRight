import pytest
from resize_right import resize
from PIL import Image
import numpy
from torchvision.transforms.functional import to_tensor
import torch
import os
from collections import defaultdict
from pathlib import Path

"""
TODO:

- [X] Add test rig
- [ ] Add targets from matlab
- [ ] Add source/target pairs for upscaling
- [ ] Test different interp methods
- [ ] Test different target sizes/aspect ratios
"""

RESOURCE_DIR = Path("tests")


def original_expected_pairs():
    """Create (original input, expected output) pairs"""
    original_suffix = "-original.png"
    prefixes = [
        f[: -len(original_suffix)]
        for f in os.listdir(RESOURCE_DIR)
        if f.endswith(original_suffix)
    ]
    params = []
    for prefix in prefixes:
        original = f"{prefix}{original_suffix}"
        expected_outputs = [
            f for f in os.listdir(RESOURCE_DIR) if f.rsplit("-", 1)[0] == prefix
        ]
        for expected_output in expected_outputs:
            params.append(pytest.param(original, expected_output, id=expected_output))
    return params


@pytest.mark.parametrize("original,expected", original_expected_pairs())
def test_equivalence(original, expected):
    """Check that torch and numpy outputs are as expected for a given input"""
    original = Image.open(RESOURCE_DIR / original)

    # Check torch path vs target image
    t_original = to_tensor(original)
    expected = Image.open(RESOURCE_DIR / expected)
    t_expected = to_tensor(expected)
    t_resized = resize(t_original, out_shape=t_expected.shape)
    assert torch.allclose(t_expected, t_resized.clip(0, 1), atol=1.0 / 255)

    # Check numpy path vs the torch path
    n_original = t_original.permute(2, 1, 0).numpy()
    n_resized = resize(n_original, out_shape=expected.size)
    assert numpy.allclose(n_resized, t_resized.permute(2, 1, 0).numpy(), atol=1.0 / 255)
