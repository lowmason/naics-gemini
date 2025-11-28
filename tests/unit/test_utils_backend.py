import types

import pytest
import torch

from naics_embedder.utils import backend

@pytest.mark.unit
def test_get_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2, raising=False)
    monkeypatch.setattr(torch.backends, 'mps', types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(torch.version, 'cuda', '12.1')

    device, precision, num = backend.get_device()

    assert device == 'cuda'
    assert precision == '16-mixed'
    assert num == 2

@pytest.mark.unit
def test_get_device_uses_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)
    monkeypatch.setattr(torch.backends, 'mps', types.SimpleNamespace(is_available=lambda: True))

    device, precision, num = backend.get_device()

    assert device == 'mps'
    assert precision == '32-true'
    assert num == 1

@pytest.mark.unit
def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)
    monkeypatch.setattr(torch.backends, 'mps', types.SimpleNamespace(is_available=lambda: False))

    device, precision, num = backend.get_device()

    assert device == 'cpu'
    assert precision == '32-true'
    assert num == 0
