import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import iwcl
from iwcl.losses import masked_cross_entropy


def release_loss(outputs, target, alpha, name):
    """Line-by-line copy of the loss of archive/release_2023/train.py (Solver.train)."""
    criterion = nn.CrossEntropyLoss()
    regularization = {"L1": nn.L1Loss(), "MSE": nn.MSELoss(), "KL": nn.KLDivLoss(), "Hubor": nn.SmoothL1Loss()}[name]
    num_imgs = len(outputs)
    mean = 0
    for output in outputs:
        mean += output
    mean /= num_imgs
    loss = 0
    for idx in range(num_imgs):
        if name == "KL":  # F.softmax(x) without dim uses dim=1 for 2-D inputs
            loss += criterion(outputs[idx], target) + regularization(F.softmax(outputs[idx], dim=1).log(), F.softmax(mean, dim=1))
        else:
            mask = target != -1
            if mask.sum() > 0:
                loss += alpha * regularization(outputs[idx], mean.detach()) + (1 - alpha) * criterion(outputs[idx], target)
            else:
                loss += alpha * regularization(outputs[idx], mean.detach())
    return loss / num_imgs


def paper_loss(logits, target, lam, reduce_classes=False):
    """Eqs. (2)-(6) of the paper written out with explicit sums."""
    n_views, batch, n_classes = logits.shape
    center = logits.mean(0).detach()                                            # Eq. (2)
    l_ic = ((logits - center) ** 2).sum() / (batch * n_views)                   # Eq. (6)
    if reduce_classes:
        l_ic = l_ic / n_classes                                                 # nn.MSELoss normalisation
    l_ce = sum(F.cross_entropy(logits[n], target, reduction="sum") for n in range(n_views)) / (batch * n_views)  # Eq. (4)
    return (1 - lam) * l_ce + lam * l_ic                                        # Eq. (3)


@pytest.mark.parametrize("num_views", [1, 2, 3, 5])
@pytest.mark.parametrize("lam", [0.7, 0.5, 0.1])
def test_matches_the_equations_of_the_paper(num_views, lam):
    torch.manual_seed(0)
    z, y = 3 * torch.randn(num_views, 8, 10), torch.randint(0, 10, (8,))
    torch.testing.assert_close(iwcl.InstanceWiseCenterLoss(alpha=lam, reduction="sum")(z, y), paper_loss(z, y, lam))
    torch.testing.assert_close(iwcl.InstanceWiseCenterLoss(alpha=lam)(z, y), paper_loss(z, y, lam, reduce_classes=True))


@pytest.mark.parametrize("name,distance", [("MSE", "mse"), ("L1", "l1"), ("Hubor", "huber")])
@pytest.mark.parametrize("num_views", [1, 2, 3, 5])
@pytest.mark.parametrize("alpha", [0.5, 0.2])
def test_matches_release_code(name, distance, num_views, alpha):
    torch.manual_seed(0)
    base = 3 * torch.randn(num_views, 8, 10)
    target = torch.randint(0, 10, (8,))
    a = base.clone().requires_grad_(True)
    b = base.clone().requires_grad_(True)
    expected = release_loss([a[k] for k in range(num_views)], target, alpha, name)
    expected.backward()
    got = iwcl.InstanceWiseCenterLoss(alpha=alpha, distance=distance)(b, target)
    got.backward()
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(b.grad, a.grad, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("num_views", [2, 3])
def test_release_kl_branch_is_twice_the_convex_loss(num_views):
    """The release KL branch used CE + KL (no lambda, no stop-gradient); the paper weights
    every distance with lambda_IC. CE + KL = 2 x the convex loss with lambda_IC = 0.5, with
    the same gradient direction (the stop-gradient does not matter for KL)."""
    torch.manual_seed(0)
    base = 3 * torch.randn(num_views, 8, 10)
    target = torch.randint(0, 10, (8,))
    a = base.clone().requires_grad_(True)
    b = base.clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KLDivLoss 'mean' reduction warning
        expected = release_loss([a[k] for k in range(num_views)], target, 0.5, "KL")
    expected.backward()
    got = iwcl.InstanceWiseCenterLoss(alpha=0.5, distance="kl")(b, target)
    got.backward()
    torch.testing.assert_close(2 * got, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(2 * b.grad, a.grad, rtol=1e-5, atol=1e-7)


def test_input_layouts_agree():
    torch.manual_seed(1)
    z = torch.randn(3, 4, 6)
    y = torch.randint(0, 6, (4,))
    loss = iwcl.InstanceWiseCenterLoss()
    ref = loss(z, y)
    torch.testing.assert_close(loss([z[0], z[1], z[2]], y), ref)
    torch.testing.assert_close(iwcl.InstanceWiseCenterLoss(num_views=3)(z.reshape(12, 6), y), ref)
    with pytest.raises(ValueError):
        loss(z.reshape(12, 6), y)  # flat input without num_views is ambiguous


def test_single_view_and_identical_views_reduce_to_weighted_ce():
    torch.manual_seed(2)
    z = torch.randn(1, 5, 7)
    y = torch.randint(0, 7, (5,))
    ce = F.cross_entropy(z[0], y)
    torch.testing.assert_close(iwcl.InstanceWiseCenterLoss(alpha=0.3)(z, y), 0.7 * ce)
    torch.testing.assert_close(iwcl.InstanceWiseCenterLoss(alpha=0.3, distance="kl")(z, y), 0.7 * ce)
    same = z.expand(4, 5, 7)  # four identical views: the centre term vanishes
    torch.testing.assert_close(iwcl.InstanceWiseCenterLoss(alpha=0.3)(same, y), 0.7 * ce)


def test_defaults_follow_the_paper():
    loss = iwcl.InstanceWiseCenterLoss()
    assert (loss.alpha, loss.distance, loss.stopgrad, loss.reduction, loss.center_on) == (0.5, "mse", True, "mean", "logits")
    assert iwcl.InstanceWiseCenterLoss(distance="L2").distance == "mse"
    assert iwcl.InstanceWiseCenterLoss(distance="Hubor").distance == "huber"  # release config spelling


def _grad(distance, stopgrad, z, y):
    x = z.clone().requires_grad_(True)
    iwcl.InstanceWiseCenterLoss(alpha=0.5, distance=distance, stopgrad=stopgrad)(x, y).backward()
    return x.grad


def test_stopgrad_only_changes_the_gradient_for_l1_and_huber():
    # the centre is the mean of the views: the gradient through it vanishes for mse and kl
    torch.manual_seed(3)
    z, y = 3 * torch.randn(3, 6, 5, dtype=torch.float64), torch.randint(0, 5, (6,))
    for distance in ("mse", "kl"):
        torch.testing.assert_close(_grad(distance, True, z, y), _grad(distance, False, z, y), rtol=0, atol=1e-12)
    for distance in ("l1", "huber"):
        assert (_grad(distance, True, z, y) - _grad(distance, False, z, y)).abs().max() > 1e-4


def test_unlabelled_images_only_enter_the_centre_term():
    torch.manual_seed(4)
    z = torch.randn(2, 6, 5)
    y = torch.tensor([0, -1, 2, -1, 4, 1])
    keep = y != -1
    center = z.mean(0)
    expected = sum(0.5 * F.mse_loss(z[k], center) + 0.5 * F.cross_entropy(z[k][keep], y[keep]) for k in range(2)) / 2
    torch.testing.assert_close(iwcl.InstanceWiseCenterLoss()(z, y), expected)
    # a batch without labels keeps only alpha * centre term, as in the release code
    y_none = torch.full((6,), -1)
    expected = sum(0.5 * F.mse_loss(z[k], center) for k in range(2)) / 2
    torch.testing.assert_close(iwcl.InstanceWiseCenterLoss()(z, y_none), expected)
    assert masked_cross_entropy(z[0], y_none).item() == 0.0


def test_center_on_features():
    torch.manual_seed(5)
    logits, feats = torch.randn(2, 4, 3), torch.randn(2, 4, 8)
    y = torch.randint(0, 3, (4,))
    loss = iwcl.InstanceWiseCenterLoss(alpha=0.4, center_on="features")
    center = feats.mean(0)
    expected = sum(0.4 * F.mse_loss(feats[k], center) + 0.6 * F.cross_entropy(logits[k], y) for k in range(2)) / 2
    torch.testing.assert_close(loss(logits, y, feats), expected)
    with pytest.raises(ValueError):
        loss(logits, y)


def test_helpers_and_validation():
    z = torch.arange(12.0).reshape(2, 3, 2)
    torch.testing.assert_close(iwcl.instance_centers(z), z.mean(0))
    assert iwcl.view_spread(z.expand(2, 3, 2) * 0 + 1).item() == 0.0
    for bad in [dict(distance="cosine"), dict(alpha=1.5), dict(center_on="pixels"), dict(distance="kl", center_on="features"), dict(reduction="none")]:
        with pytest.raises(ValueError):
            iwcl.InstanceWiseCenterLoss(**bad)
    with pytest.raises(ValueError):
        iwcl.InstanceWiseCenterLoss()(torch.randn(2, 4, 3), torch.zeros(5, dtype=torch.long))
