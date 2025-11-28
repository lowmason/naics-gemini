from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from naics_embedder.cli import app as cli_app
from naics_embedder.cli.commands import training
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import CheckpointInfo, HardwareInfo
from naics_embedder.utils.validation import ValidationResult

@pytest.fixture
def cli_runner():
    return CliRunner()

@pytest.fixture
def training_env(monkeypatch, tmp_path):
    context = SimpleNamespace()
    context.checkpoint_info = CheckpointInfo(path=None, is_same_stage=False, exists=False)
    context.validation_result = ValidationResult.success()
    context.save_summary_calls = []
    context.fail_during_fit = False
    context.trainer = None

    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    monkeypatch.setattr(training, 'detect_hardware', lambda log_info=False: hardware)

    def build_cfg():
        cfg = Config()
        cfg.experiment_name = 'cli-test'
        outputs_dir = tmp_path / 'outputs'
        checkpoints_dir = tmp_path / 'checkpoints'
        outputs_dir.mkdir(exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
        cfg.dirs.output_dir = str(outputs_dir)
        cfg.dirs.checkpoint_dir = str(checkpoints_dir)
        desc_path = tmp_path / 'descriptions.parquet'
        desc_path.write_text('data')
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir(exist_ok=True)
        cfg.data_loader.streaming.descriptions_parquet = str(desc_path)
        cfg.data_loader.streaming.triplets_parquet = str(triplets_dir)
        cfg.training.trainer.max_epochs = 1
        cfg.training.trainer.devices = 1
        cfg.training.trainer.log_every_n_steps = 1
        cfg.training.trainer.val_check_interval = 1.0
        cfg.training.trainer.gradient_clip_val = 0.5
        cfg.training.trainer.accumulate_grad_batches = 1
        cfg.loss.__dict__['base_margin'] = 1.0
        return cfg

    monkeypatch.setattr(training.Config, 'from_yaml', classmethod(lambda cls, path: build_cfg()))
    monkeypatch.setattr(training, 'validate_training_config', lambda cfg: context.validation_result)

    original_override = Config.override

    def override_with_margin(self, overrides):
        new_cfg = original_override(self, overrides)
        new_cfg.loss.__dict__['base_margin'] = 1.0
        return new_cfg

    monkeypatch.setattr(Config, 'override', override_with_margin, raising=False)

    def fake_resolve(ckpt_path, checkpoint_dir, experiment_name):
        context.resolve_args = (ckpt_path, str(checkpoint_dir), experiment_name)
        return context.checkpoint_info

    monkeypatch.setattr(training, 'resolve_checkpoint', fake_resolve)

    class DummyDataModule:

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(training, 'NAICSDataModule', DummyDataModule)

    class DummyModel:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded_from_ckpt = False

        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            instance = cls(**kwargs)
            instance.loaded_from_ckpt = True
            instance.ckpt_path = path
            return instance

    monkeypatch.setattr(training, 'NAICSContrastiveModel', DummyModel)

    class DummyTrainer:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_calls = []
            context.trainer = self

        def fit(self, model, datamodule, ckpt_path=None):
            if context.fail_during_fit:
                raise RuntimeError('boom')
            self.fit_calls.append(
                {
                    'model': model,
                    'datamodule': datamodule,
                    'ckpt_path': ckpt_path
                }
            )

    monkeypatch.setattr(training.pyl, 'Trainer', DummyTrainer)

    def fake_save_summary(**kwargs):
        context.save_summary_calls.append(kwargs)
        return {'json': str(tmp_path / 'summary.json')}

    monkeypatch.setattr(training, 'save_training_summary', fake_save_summary)
    monkeypatch.setattr(training.typer, 'confirm', lambda *_, **__: False)

    return context

@pytest.mark.unit
def test_cli_train_runs_with_defaults(cli_runner, training_env):
    result = cli_runner.invoke(cli_app, ['train'], catch_exceptions=False)

    assert result.exit_code == 0
    assert training_env.trainer is not None
    assert training_env.trainer.fit_calls[0]['ckpt_path'] is None
    assert training_env.save_summary_calls

@pytest.mark.unit
def test_cli_train_applies_overrides(cli_runner, training_env, monkeypatch):
    captured = {}

    def fake_parse(overrides):
        captured['overrides'] = overrides
        return {'training.learning_rate': 0.5}, []

    monkeypatch.setattr(training, 'parse_config_overrides', fake_parse)

    result = cli_runner.invoke(
        cli_app, ['train', 'training.learning_rate=0.5'], catch_exceptions=False
    )

    assert result.exit_code == 0
    assert captured['overrides'] == ['training.learning_rate=0.5']

@pytest.mark.unit
def test_training_workflow_validation_failure(training_env):
    training_env.validation_result = ValidationResult(valid=False, errors=['boom'], warnings=[])

    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=False)

    assert excinfo.value.exit_code == 1

@pytest.mark.unit
def test_training_checkpoint_resume_passes_ckpt(training_env):
    training_env.checkpoint_info = CheckpointInfo(path='foo.ckpt', is_same_stage=True, exists=True)

    training.train(ckpt_path='last', skip_validation=True)

    assert training_env.trainer.fit_calls[0]['ckpt_path'] == 'foo.ckpt'

@pytest.mark.unit
def test_training_error_handling_exits(training_env):
    training_env.fail_during_fit = True

    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=True)

    assert excinfo.value.exit_code == 1
