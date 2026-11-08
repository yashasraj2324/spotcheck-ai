import pytest
import torch

from skin_lesion.models.factory import build_model

NUM_CLASSES = 7


@pytest.mark.parametrize("name", ["baseline", "hybrid"])
def test_output_shape(tiny_cfg, name):
    tiny_cfg["model"]["name"] = name
    model = build_model(tiny_cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("name", ["baseline", "hybrid"])
def test_param_groups_partition(tiny_cfg, name):
    tiny_cfg["model"]["name"] = name
    model = build_model(tiny_cfg)
    ids = [id(p) for p in model.backbone_parameters() + model.head_parameters()]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))


def test_unknown_model_raises(tiny_cfg):
    tiny_cfg["model"]["name"] = "nope"
    with pytest.raises(ValueError):
        build_model(tiny_cfg)


def test_hybrid_rejects_wrong_input_size(tiny_cfg):
    tiny_cfg["model"]["name"] = "hybrid"
    model = build_model(tiny_cfg).eval()
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 128, 128))
