"""Tests for selection head."""

import pytest

torch = pytest.importorskip("torch")

from openfugu.router.head import SelectionHead


def test_head_output_shape():
    head = SelectionHead(hidden_dim=128, num_workers=3)
    h = torch.randn(128)
    logits = head(h)
    assert logits.shape == (1, 3)


def test_head_param_count():
    head = SelectionHead(hidden_dim=1024, num_workers=3)
    # 1024*3 + 3 bias = 3075
    assert head.param_count() == 1024 * 3 + 3


def test_batch_hidden_state():
    head = SelectionHead(hidden_dim=64, num_workers=2)
    h = torch.randn(4, 64)
    logits = head(h)
    assert logits.shape == (4, 2)
