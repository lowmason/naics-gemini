from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from naics_embedder.utils.config import Config
from naics_embedder.utils.training import (
    HardwareInfo,
    TrainingResult,
    collect_training_result,
    create_trainer,
    detect_hardware,
    get_gpu_memory_info,
    parse_config_overrides,
    resolve_checkpoint,
    save_training_summary,
)

def _build_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.experiment_name = 'unit-test'
    cfg.dirs.output_dir = str(tmp_path / 'outputs')
    cfg.dirs.checkpoint_dir = str(tmp_path / 'checkpoints')
    Path(cfg.dirs.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.dirs.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    cfg.training.trainer.max_epochs = 1
    cfg.training.trainer.devices = 1
    cfg.training.trainer.gradient_clip_val = 0.5
    cfg.training.trainer.accumulate_grad_batches = 1
    cfg.training.trainer.log_every_n_steps = 1
    cfg.training.trainer.val_check_interval = 1.0
    return cfg

@pytest.mark.unit
def test_detect_hardware_cuda_collects_gpu_memory(monkeypatch):
    monkeypatch.setattr(
        'naics_embedder.utils.training.get_device',
        lambda log_info=False: ('cuda', '16-mixed', 2),
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True, raising=False)
    monkeypatch.setattr(
        'naics_embedder.utils.training.get_gpu_memory_info',
        lambda: {'total_gb': 24.0},
    )

    info = detect_hardware(log_info=True)

    assert info.accelerator == 'cuda'
    assert info.precision == '16-mixed'
    assert info.gpu_memory == {'total_gb': 24.0}

@pytest.mark.unit
def test_detect_hardware_cpu_fallback(monkeypatch):
    monkeypatch.setattr(
        'naics_embedder.utils.training.get_device',
        lambda log_info=False: ('cpu', '32-true', 1),
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)

    info = detect_hardware()

    assert info.accelerator == 'cpu'
    assert info.gpu_memory is None

@pytest.mark.unit
def test_get_gpu_memory_info_returns_stats(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        lambda device: SimpleNamespace(total_memory=8 * 1024**3),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 2 * 1024**3, raising=False)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 1 * 1024**3, raising=False)

    stats = get_gpu_memory_info()

    assert stats is not None
    assert pytest.approx(stats['total_gb'], rel=1e-3) == 8.0
    assert pytest.approx(stats['reserved_gb'], rel=1e-3) == 2.0
    assert pytest.approx(stats['allocated_gb'], rel=1e-3) == 1.0
    assert pytest.approx(stats['free_gb'], rel=1e-3) == 6.0

@pytest.mark.unit
def test_get_gpu_memory_info_none_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)
    assert get_gpu_memory_info() is None

@pytest.mark.unit
def test_parse_config_overrides_handles_invalid_entries():
    overrides = ['training.learning_rate=1e-4', 'bad_override', 'trainer.devices=2']
    parsed, invalid = parse_config_overrides(overrides)

    assert parsed['training.learning_rate'] == pytest.approx(1e-4)
    assert parsed['trainer.devices'] == 2
    assert invalid == ['bad_override']

@pytest.mark.unit
def test_resolve_checkpoint_last_keyword(tmp_path):
    experiment_dir = tmp_path / 'exp'
    experiment_dir.mkdir()
    last_ckpt = experiment_dir / 'last.ckpt'
    last_ckpt.write_text('checkpoint')

    info = resolve_checkpoint('last', tmp_path, 'exp')
    assert info.exists
    assert info.is_same_stage
    assert info.path == str(last_ckpt)

@pytest.mark.unit
def test_resolve_checkpoint_explicit_path(tmp_path):
    ckpt = tmp_path / 'custom.ckpt'
    ckpt.write_text('ckpt')

    info = resolve_checkpoint(str(ckpt), tmp_path, 'exp')
    assert info.exists
    assert info.path == str(ckpt.resolve())

@pytest.mark.unit
def test_resolve_checkpoint_missing_path(tmp_path):
    info = resolve_checkpoint('missing.ckpt', tmp_path, 'exp')
    assert info.exists is False
    assert info.path is None

@pytest.mark.unit
def test_create_trainer_uses_cpu_defaults(tmp_path):
    cfg = _build_config(tmp_path)
    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer, ckpt_cb, es_cb = create_trainer(cfg, hardware, checkpoint_dir)

    assert ckpt_cb.monitor == 'val/contrastive_loss'
    assert es_cb.patience == 3
    assert trainer.max_epochs == cfg.training.trainer.max_epochs
    assert trainer.logger is not None

@pytest.mark.unit
def test_create_trainer_multi_gpu_uses_ddp(monkeypatch, tmp_path):
    cfg = _build_config(tmp_path)
    cfg.training.trainer.devices = 2
    hardware = HardwareInfo(accelerator='cuda', precision='16-mixed', num_devices=2)
    checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    class DummyTrainer:

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.kwargs = kwargs

    class DummyStrategy:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr('naics_embedder.utils.training.pyl.Trainer', DummyTrainer)
    monkeypatch.setattr('pytorch_lightning.strategies.DDPStrategy', DummyStrategy)

    trainer, _, _ = create_trainer(cfg, hardware, checkpoint_dir)
    assert isinstance(captured['strategy'], DummyStrategy)
    assert captured['devices'] == 2
    assert isinstance(trainer, DummyTrainer)

@pytest.mark.unit
def test_collect_training_result(tmp_path):
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path), filename='model-{epoch:02d}')
    ckpt_cb.best_model_path = str(tmp_path / 'best.ckpt')
    early_stopping = EarlyStopping(monitor='val/contrastive_loss')
    early_stopping.stopped_epoch = 4
    early_stopping.best_score = torch.tensor(0.123)

    result = collect_training_result(ckpt_cb, early_stopping, config_path='conf.yaml')

    assert result.best_checkpoint_path == str(tmp_path / 'best.ckpt')
    assert result.last_checkpoint_path == str(tmp_path / 'last.ckpt')
    assert result.config_path == 'conf.yaml'
    assert result.early_stopped

@pytest.mark.unit
def test_save_training_summary_writes_files(tmp_path):
    result = TrainingResult(
        best_checkpoint_path='best.ckpt',
        last_checkpoint_path='last.ckpt',
        config_path='config.yaml',
        best_loss=0.42,
        stopped_epoch=5,
        early_stopped=True,
        metrics={'best_val_loss': 0.42},
    )
    cfg = Config()
    hw = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)

    paths = save_training_summary(result, cfg, hw, tmp_path, format='both')

    assert 'yaml' in paths and Path(paths['yaml']).exists()
    assert 'json' in paths and Path(paths['json']).exists()
