import pytest
import torch

from iwcl.models import MODELS, build_model

FEATURE_DIMS = {"wideresnet": 640, "resnet18": 512, "resnext": 1024, "densenet": 342, "shakeshake": 128}

# (archived file, constructor as called by archive/release_2023/train.py, unused heads of the
# release model, build_model arguments of the same network)
RELEASE = {
    "wideresnet": ("models/WideResNet.py", lambda m, c: m.WideResNet(depth=28, num_classes=c, widen_factor=10, drop_rate=0.0), ("fc2", "fc2_sep"), {}),
    "resnet18": ("models/resnet.py", lambda m, c: m.ResNet18(num_classes=c), ("linear2",), {}),
    "resnext": ("models/resnext.py", lambda m, c: m.CifarResNeXt(num_classes=c), (), {"resnext_base_width": 64}),
    "densenet": ("models/densenet.py", lambda m, c: m.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=c), (), {}),
    "shakeshake": ("models/shakeshake.py", lambda m, c: m.ShakeShake(input_shape=(1, 3, 32, 32), n_classes=c, base_channels=32, depth=26), (), {}),
}


@pytest.mark.parametrize("name", MODELS)
def test_shapes_and_feature_dim(name):
    torch.manual_seed(0)
    model = build_model(name, num_classes=7).eval()
    logits, features = model(torch.rand(2, 3, 32, 32))
    assert logits.shape == (2, 7)
    assert features.shape == (2, FEATURE_DIMS[name]) == (2, model.feature_dim)


@pytest.mark.parametrize("name", MODELS)
def test_same_network_as_release_code(name, load_release):
    """Release weights load into the new model and give the same outputs."""
    path, ctor, unused, kwargs = RELEASE[name]
    module = load_release(path)
    torch.manual_seed(0)
    reference = ctor(module, 10).eval()
    model = build_model(name, num_classes=10, **kwargs).eval()
    result = model.load_state_dict(reference.state_dict(), strict=False)
    assert result.missing_keys == []
    assert sorted({k.split(".")[0] for k in result.unexpected_keys}) == sorted(unused)
    n_unused = sum(p.numel() for n, p in reference.named_parameters() if n.split(".")[0] in unused)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters()) - n_unused

    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        ref_logits, ref_features = reference(x)
        logits, features = model(x)
    torch.testing.assert_close(logits, ref_logits)
    torch.testing.assert_close(features, ref_features.reshape(2, -1))


def test_shakeshake_train_mode_matches_release(load_release):
    module = load_release("models/shakeshake.py")
    reference = RELEASE["shakeshake"][1](module, 10).train()
    model = build_model("shakeshake", num_classes=10).train()
    model.load_state_dict(reference.state_dict())
    x = torch.rand(4, 3, 32, 32)
    torch.manual_seed(3)
    ref_logits, _ = reference(x)
    torch.manual_seed(3)
    logits, _ = model(x)
    torch.testing.assert_close(logits, ref_logits)


def test_densenet_returns_log_probabilities_like_the_release_model():
    model = build_model("densenet", num_classes=10).eval()
    with torch.no_grad():
        out, _ = model(torch.rand(2, 3, 32, 32))
    torch.testing.assert_close(out.exp().sum(1), torch.ones(2))


def test_resnext_is_8x32d_by_default_like_the_paper():
    model = build_model("resnext", num_classes=10)
    widths = [getattr(model, f"stage_{i}")[0].conv_reduce.out_channels for i in (1, 2, 3)]
    assert widths == [8 * 32, 8 * 64, 8 * 128]  # cardinality 8 x bottleneck width 32 (doubling per stage)
    assert model.stage_1[0].conv_conv.groups == 8
