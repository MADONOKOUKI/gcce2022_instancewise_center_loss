import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import iwcl
from iwcl.baselines import augmix_jsd, center_loss_term, contrastive_center_term


def _data(num_views=2, batch=6, classes=5, dim=4, seed=0):
    torch.manual_seed(seed)
    return torch.randn(num_views, batch, classes), torch.randn(num_views, batch, dim), torch.randint(0, classes, (batch,))


def test_baseline_is_mean_cross_entropy_over_views():
    logits, _, y = _data(num_views=3)
    expected = sum(F.cross_entropy(z, y) for z in logits) / 3
    torch.testing.assert_close(iwcl.MultiViewCrossEntropy()(logits, y), expected)


def test_center_loss_matches_reference_implementation(load_release):
    module = load_release("loss/center_loss.py")
    reference = module.CenterLoss(num_classes=5, feat_dim=4, use_gpu=False)
    logits, feats, y = _data()
    ours = iwcl.CenterLoss(num_classes=5, feat_dim=4)
    with torch.no_grad():
        ours.centers.copy_(reference.centers)
    torch.testing.assert_close(center_loss_term(feats[0], y, ours.centers), reference(feats[0], y))
    # objective of the release loop: CE + 0.1 * center loss per view, averaged over views
    expected = sum(F.cross_entropy(logits[k], y) + 0.1 * reference(feats[k], y) for k in range(2)) / 2
    torch.testing.assert_close(ours(logits, y, feats), expected)


def test_center_gradients_are_rescaled_by_inverse_weight():
    logits, feats, y = _data()
    loss_fn = iwcl.CenterLoss(num_classes=5, feat_dim=4, weight=0.1)
    loss_fn(logits, y, feats).backward()
    raw = loss_fn.centers.grad.clone()
    loss_fn.after_backward()
    torch.testing.assert_close(loss_fn.centers.grad, raw * 10)
    assert any(p is loss_fn.centers for p in loss_fn.parameters())


def test_contrastive_center_loss_matches_release_forward(load_release):
    module = load_release("loss/contrastive_center_loss.py")
    logits, feats, y = _data()
    ours = iwcl.ContrastiveCenterLoss(num_classes=5, feat_dim=4)
    # the release constructor calls .cuda(); build the object without it and reuse its forward
    reference = object.__new__(module.ContrastiveCenterLoss)
    nn.Module.__init__(reference)
    reference.num_classes, reference.lambda_c, reference.centers = 5, 1.0, ours.centers.clone()
    torch.testing.assert_close(contrastive_center_term(feats[0], y, ours.centers), reference(feats[0], y))
    expected = sum(F.cross_entropy(logits[k], y) + 0.1 * reference(feats[k], y) for k in range(2)) / 2
    torch.testing.assert_close(ours(logits, y, feats), expected)


def test_contrastive_centers_fixed_by_default_like_the_release_code():
    # the release code created the centres as nn.Parameter(...).cuda(); an operation on a
    # Parameter returns a plain tensor, so they were never registered or optimised:
    assert not isinstance(nn.Parameter(torch.randn(2, 3)).double(), nn.Parameter)
    fixed = iwcl.ContrastiveCenterLoss(num_classes=5, feat_dim=4)
    assert list(fixed.parameters()) == [] and "centers" in dict(fixed.named_buffers())
    learnable = iwcl.ContrastiveCenterLoss(num_classes=5, feat_dim=4, learnable_centers=True)
    assert [n for n, _ in learnable.named_parameters()] == ["centers"]


def test_triplet_loss_matches_release_formula():
    logits, _, y = _data(num_views=3)
    triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    expected = 0
    for idx in range(3):
        other = logits[(idx + 1) % 3]
        pos = other.size(0) - 1
        negative = torch.cat([other[pos - 1:], other[: pos - 1]], dim=0)  # release code
        expected = expected + F.cross_entropy(logits[idx], y) + 0.1 * triplet(logits[idx], other, negative)
    torch.testing.assert_close(iwcl.InstanceTripletLoss()(logits, y), expected / 3)
    with pytest.raises(ValueError):
        iwcl.InstanceTripletLoss()(logits[:1], y)


def test_triplet_loss_on_features():
    logits, feats, y = _data(num_views=2)
    triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    expected = sum(
        F.cross_entropy(logits[k], y) + 0.1 * triplet(feats[k], feats[1 - k], torch.roll(feats[1 - k], 2, 0)) for k in range(2)
    ) / 2
    torch.testing.assert_close(iwcl.InstanceTripletLoss(on="features")(logits, y, feats), expected)


def test_augmix_matches_release_accumulation():
    logits, _, y = _data(num_views=3)
    p_clean, p_aug1, p_aug2 = (F.softmax(z, dim=1) for z in logits)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1).log()
    loss = 0
    for _ in range(3):  # the release loop added the same terms num_imgs (= 3) times, then divided by 3
        loss = loss + 12 * (
            F.kl_div(p_mixture, p_clean, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug1, reduction="batchmean")
            + F.kl_div(p_mixture, p_aug2, reduction="batchmean")
        ) / 3.0
        loss = loss + F.cross_entropy(logits[0], y)
    torch.testing.assert_close(iwcl.AugMixJSDLoss()(logits, y), loss / 3)
    assert augmix_jsd(logits[0], logits[0], logits[0]).abs().item() < 1e-6
    with pytest.raises(ValueError):
        iwcl.AugMixJSDLoss()(logits[:2], y)


def test_augmix_jsd_gradient_stays_finite_when_a_probability_underflows():
    z = torch.tensor([[0.0, 120.0, 0.0]], requires_grad=True)   # softmax underflows to exactly 0
    w = torch.tensor([[0.0, 0.0, 1.0]])
    augmix_jsd(z, w, w).backward()
    assert torch.isfinite(z.grad).all()
    # the original formulation F.kl_div(log M, softmax(z)) gives NaN here
    z2 = z.detach().clone().requires_grad_(True)
    p = [F.softmax(t, dim=1) for t in (z2, w, w)]
    log_m = torch.clamp(sum(p) / 3.0, 1e-7, 1).log()
    sum(F.kl_div(log_m, q, reduction="batchmean") for q in p).backward()
    assert not torch.isfinite(z2.grad).all()


@pytest.mark.parametrize("method", iwcl.METHODS + ("augmix",))
def test_build_criterion_runs_every_method(method):
    num_views = 3 if method in ("jsd", "augmix") else 2
    logits, feats, y = _data(num_views=num_views)
    logits.requires_grad_(True)
    criterion = iwcl.build_criterion(method, num_classes=5, feat_dim=4)
    loss = criterion(logits, y, feats)
    loss.backward()
    getattr(criterion, "after_backward", lambda: None)()
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert logits.grad is not None


def test_method_names_and_default_weights():
    assert isinstance(iwcl.build_criterion("augmix"), iwcl.AugMixJSDLoss)  # release config name
    assert iwcl.build_criterion("jsd").weight == 12.0
    assert iwcl.build_criterion("triplet_loss").weight == 0.1
    assert iwcl.build_criterion("center_loss", num_classes=5, feat_dim=4).weight == 0.1
    assert iwcl.build_criterion("contrastive_center_loss", num_classes=5, feat_dim=4).weight == 0.1
