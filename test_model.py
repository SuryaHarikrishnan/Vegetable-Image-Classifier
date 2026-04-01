import pytest
import os
from PIL import Image

# ------------------------------
# 1️⃣ Create dummy dataset if missing
# ------------------------------
def create_dummy_dataset():
    for split in ["train", "val", "test"]:
        for cls in ["apple", "carrot"]:
            path = os.path.join("dataset", "labeled", split, cls)
            os.makedirs(path, exist_ok=True)
            img_path = os.path.join(path, "img1.jpg")
            if not os.path.exists(img_path):
                Image.new("RGB", (128,128)).save(img_path)

create_dummy_dataset()  # Make sure folders exist before importing model

# ------------------------------
# 2️⃣ Import model AFTER dataset exists
# ------------------------------
from model import train_model
import torch.nn as nn

# ------------------------------
# 3️⃣ Use fixture for output
# ------------------------------
@pytest.fixture(scope="module")
def output():
    # quick_test=True runs 1 epoch for fast tests
    return train_model(quick_test=True)

# ------------------------------
# 4️⃣ Tests
# ------------------------------
def test_output_is_dict(output):
    assert isinstance(output, dict), "train_model should return a dict"

def test_output_format(output):
    keys = ["model", "test_acc", "loaders", "history"]
    assert set(output.keys()) == set(keys), "output dict has wrong keys"

    assert isinstance(output["model"], nn.Module), "model key must be a PyTorch model"
    assert isinstance(output["test_acc"], float), "test_acc key must be a float"
    assert isinstance(output["loaders"], tuple), "loaders key must be a tuple"
    assert isinstance(output["history"], dict), "history key must be a dictionary"
