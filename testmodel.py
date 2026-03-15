import pytest
from model import train_model
import torch.nn as nn

output = train_model()

def test_output_is_dict():
    assert isinstance(output, dict)

def test_output_format():

    assert set(output.keys()) == set(
        ["model", "test_acc", "loaders", "history"]
    )

    assert isinstance(output.get("model"), nn.Module)

    assert isinstance(output.get("test_acc"), float)

    assert isinstance(output.get("loaders"), tuple)

    assert isinstance(output.get("history"), dict)
