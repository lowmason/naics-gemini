import pytest
from typer.testing import CliRunner

from naics_embedder.cli.commands import data as data_cli
from naics_embedder.cli.commands import tools as tools_cli

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture(autouse=True)
def no_logging(monkeypatch):
    monkeypatch.setattr(data_cli, 'configure_logging', lambda *_, **__: None)
    monkeypatch.setattr(tools_cli, 'configure_logging', lambda *_, **__: None)

def test_data_preprocess_invokes_download(monkeypatch, runner):
    called = {}
    monkeypatch.setattr(
        data_cli, 'download_preprocess_data', lambda: called.setdefault('preprocess', True)
    )

    result = runner.invoke(data_cli.app, ['preprocess'])

    assert result.exit_code == 0
    assert called['preprocess']

def test_data_all_runs_pipeline_in_order(monkeypatch, runner):
    order = []
    monkeypatch.setattr(data_cli, 'download_preprocess_data', lambda: order.append('preprocess'))
    monkeypatch.setattr(data_cli, 'calculate_pairwise_relations', lambda: order.append('relations'))
    monkeypatch.setattr(data_cli, 'calculate_pairwise_distances', lambda: order.append('distances'))
    monkeypatch.setattr(data_cli, 'generate_training_triplets', lambda: order.append('triplets'))

    result = runner.invoke(data_cli.app, ['all'])

    assert result.exit_code == 0
    assert order == ['preprocess', 'relations', 'distances', 'triplets']

def test_tools_config_passes_config_path(monkeypatch, runner, tmp_path):
    captured = {}
    monkeypatch.setattr(
        tools_cli, 'show_current_config', lambda cfg_path: captured.setdefault('path', cfg_path)
    )
    config_path = tmp_path / 'custom.yaml'
    config_path.write_text('foo: bar')

    result = runner.invoke(tools_cli.app, ['config', '--config', str(config_path)])

    assert result.exit_code == 0
    assert captured['path'] == str(config_path)

def test_tools_visualize_handles_exception(monkeypatch, runner):

    def boom(**_kwargs):
        raise RuntimeError('boom')

    monkeypatch.setattr(tools_cli, 'visualize_metrics', boom)

    result = runner.invoke(tools_cli.app, ['visualize'])

    assert result.exit_code == 1
    assert 'Error' in result.output

def test_tools_investigate_success(monkeypatch, runner):
    monkeypatch.setattr(
        tools_cli,
        'investigate_hierarchy',
        lambda **_: {
            'reason': 'ok',
            'suggestion': 'none'
        },
    )

    result = runner.invoke(tools_cli.app, ['investigate'])

    assert result.exit_code == 0
    assert 'Investigation complete' in result.output

def test_verify_stage4_failure_sets_exit_code(monkeypatch, runner):

    def fake_verify(*_args, **_kwargs):
        return {
            'pre': {
                'metric': 0.8
            },
            'post': {
                'metric': 0.7
            },
            'delta': {
                'metric': -0.1
            },
            'checks': {
                'cophenetic': False
            },
            'passed': False,
        }

    monkeypatch.setattr(tools_cli, 'verify_stage4', fake_verify)

    result = runner.invoke(tools_cli.app, ['verify-stage4'])

    assert result.exit_code == 1
    assert 'Verification failed' in result.output or 'failed thresholds' in result.output
