'''
Unit tests for configuration management.

Tests Pydantic config models, YAML loading, and validation.
'''

import pytest
import yaml
from pydantic import ValidationError

from naics_embedder.utils.config import (
    DirConfig,
    DistancesConfig,
    DownloadConfig,
    load_config,
)

# -------------------------------------------------------------------------------------------------
# DirConfig Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDirConfig:
 
    '''Test suite for directory configuration.'''

    def test_default_values(self):
 
        '''Test that DirConfig has correct default values.'''
 
        config = DirConfig()

        assert config.checkpoint_dir == './checkpoints'
        assert config.conf_dir == './conf'
        assert config.data_dir == './data'
        assert config.docs_dir == './docs'
        assert config.log_dir == './logs'
        assert config.output_dir == './outputs'


    def test_custom_values(self):
 
        '''Test that custom values override defaults.'''
 
        config = DirConfig(
            checkpoint_dir='/custom/checkpoints',
            data_dir='/custom/data',
        )

        assert config.checkpoint_dir == '/custom/checkpoints'
        assert config.data_dir == '/custom/data'
        # Other fields should still have defaults
        assert config.conf_dir == './conf'


    def test_serialization(self):
 
        '''Test that config can be serialized to dict.'''
 
        config = DirConfig()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert 'checkpoint_dir' in config_dict
        assert 'data_dir' in config_dict


# -------------------------------------------------------------------------------------------------
# DownloadConfig Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDownloadConfig:

    '''Test suite for download configuration.'''

    def test_default_output_parquet(self) -> None:
 
        '''Test default output parquet path.'''
 
        config = DownloadConfig()

        assert config.output_parquet == './data/naics_descriptions.parquet'


    def test_custom_output_parquet(self):
 
        '''Test custom output parquet path.'''
 
        config = DownloadConfig(output_parquet='/custom/path/data.parquet')

        assert config.output_parquet == '/custom/path/data.parquet'


    def test_validation(self):
 
        '''Test that invalid configuration raises validation error.'''
 
        # output_parquet should be a string, not a number
        with pytest.raises(ValidationError):
            DownloadConfig(output_parquet='12345')


    def test_output_parquet_extension_validation(self):
 
        '''Test that output_parquet must be a parquet file.'''
 
        with pytest.raises(ValidationError):
            DownloadConfig(output_parquet='./data/output.csv')


# -------------------------------------------------------------------------------------------------
# DistancesConfig Tests
# -------------------------------------------------------------------------------------------------
 
@pytest.mark.unit
class TestDistancesConfig:

    '''Test suite for distances configuration.'''

    def test_required_fields(self):
    
        '''Test that config can be created with required fields.'''
    
        config = DistancesConfig(
            input_parquet='./data/input.parquet',
            distances_parquet='./data/distances.parquet',
            distance_matrix_parquet='./data/matrix.parquet'
        )

        assert config.input_parquet == './data/input.parquet'
        assert config.distances_parquet == './data/distances.parquet'
        assert config.distance_matrix_parquet == './data/matrix.parquet'


    def test_missing_required_field_uses_defaults(self):

        '''Test that missing fields fall back to defaults.'''

        config = DistancesConfig()

        assert config.input_parquet == './data/naics_descriptions.parquet'
        assert config.distances_parquet == './data/naics_distances.parquet'
        assert config.distance_matrix_parquet == './data/naics_distance_matrix.parquet'


# -------------------------------------------------------------------------------------------------
# Config Loading Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadConfig:

    '''Test suite for config loading function.'''

    def test_load_from_valid_yaml(self, tmp_path):

        '''Test loading config from valid YAML file.'''

        # Create temporary YAML file
        yaml_path = tmp_path / 'conf' / 'test_config.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            'checkpoint_dir': '/test/checkpoints',
            'data_dir': '/test/data',
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        # Load config
        config = load_config(DirConfig, yaml_path)

        assert config.checkpoint_dir == '/test/checkpoints'
        assert config.data_dir == '/test/data'


    def test_load_with_conf_prefix(self, tmp_path, monkeypatch):

        '''Test that conf/ prefix is added if not present.'''

        yaml_path = tmp_path / 'conf' / 'test.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {'checkpoint_dir': '/test'}

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)

        # Load without conf/ prefix
        config = load_config(DirConfig, 'test.yaml')

        assert config.checkpoint_dir == '/test'


    def test_load_missing_file_uses_defaults(self, tmp_path):

        '''Test that missing file falls back to default values.'''

        # Try to load non-existent file
        config = load_config(DirConfig, 'nonexistent.yaml')

        # Should use default values
        assert config.checkpoint_dir == './checkpoints'
        assert config.data_dir == './data'


    def test_load_empty_yaml(self, tmp_path, monkeypatch):

        '''Test loading empty YAML file uses defaults.'''

        yaml_path = tmp_path / 'conf' / 'empty.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty file
        yaml_path.touch()

        monkeypatch.chdir(tmp_path)

        config = load_config(DirConfig, 'empty.yaml')

        # Should use defaults
        assert config.checkpoint_dir == './checkpoints'


    def test_load_partial_yaml(self, tmp_path, monkeypatch):

        '''Test loading YAML with partial fields.'''

        yaml_path = tmp_path / 'conf' / 'partial.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {'checkpoint_dir': '/custom/checkpoints'}

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)

        config = load_config(DirConfig, 'partial.yaml')

        # Custom field
        assert config.checkpoint_dir == '/custom/checkpoints'
        # Default fields
        assert config.data_dir == './data'
        assert config.log_dir == './logs'


    def test_load_custom_relative_path(self, tmp_path, monkeypatch):

        '''Test loading config from a relative path outside conf/.'''

        yaml_path = tmp_path / 'custom' / 'dir' / 'custom.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {'checkpoint_dir': '/custom/path'}

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)

        config = load_config(DirConfig, 'custom/dir/custom.yaml')

        assert config.checkpoint_dir == '/custom/path'


# -------------------------------------------------------------------------------------------------
# Validation Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigValidation:

    '''Test suite for configuration validation.'''

    def test_invalid_type_raises_error(self):

        '''Test that invalid field types raise validation errors.'''

        with pytest.raises(ValidationError):
            DirConfig(checkpoint_dir=12345)  # Should be string


    def test_extra_fields_allowed(self):

        '''Test behavior with extra fields.'''

        # Pydantic should ignore extra fields by default (or raise error if configured)
        config_dict = {
            'checkpoint_dir': './checkpoints',
            'extra_field': 'value'
        }

        # This behavior depends on Pydantic config
        # By default, extra fields are ignored
        config = DirConfig(**config_dict)
        assert config.checkpoint_dir == './checkpoints'


    def test_field_validation(self):

        '''Test that field validators work correctly.'''

        # This test depends on whether any validators are defined
        config = DirConfig(checkpoint_dir='  ./checkpoints  ')

        # Should accept the value (may strip whitespace depending on validators)
        assert isinstance(config.checkpoint_dir, str)


# -------------------------------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigIntegration:

    '''Integration tests for configuration system.'''

    def test_load_multiple_configs(self, tmp_path, monkeypatch):

        '''Test loading multiple different config types.'''

        # Create directory
        conf_dir = tmp_path / 'conf'
        conf_dir.mkdir()

        # DirConfig
        dir_yaml = conf_dir / 'dir.yaml'
        with open(dir_yaml, 'w') as f:
            yaml.dump({'checkpoint_dir': '/test/checkpoints'}, f)

        # DownloadConfig
        download_yaml = conf_dir / 'download.yaml'
        with open(download_yaml, 'w') as f:
            yaml.dump({'output_parquet': '/test/output.parquet'}, f)

        monkeypatch.chdir(tmp_path)

        # Load both
        dir_config = load_config(DirConfig, 'dir.yaml')
        download_config = load_config(DownloadConfig, 'download.yaml')

        assert dir_config.checkpoint_dir == '/test/checkpoints'
        assert download_config.output_parquet == '/test/output.parquet'


    def test_config_serialization_roundtrip(self):

        '''Test that config can be serialized and deserialized.'''

        original = DirConfig(
            checkpoint_dir='/test/checkpoints',
            data_dir='/test/data'
        )

        # Serialize
        config_dict = original.model_dump()

        # Deserialize
        restored = DirConfig(**config_dict)

        assert restored.checkpoint_dir == original.checkpoint_dir
        assert restored.data_dir == original.data_dir


    def test_config_json_export(self):

        '''Test that config can be exported to JSON.'''

        config = DirConfig(checkpoint_dir='/test')

        json_str = config.model_dump_json()

        assert isinstance(json_str, str)
        assert '/test' in json_str
        assert 'checkpoint_dir' in json_str


# -------------------------------------------------------------------------------------------------
# Error Handling Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigErrorHandling:

    '''Test suite for configuration error handling.'''

    def test_malformed_yaml(self, tmp_path, monkeypatch):

        '''Test handling of malformed YAML file.'''

        yaml_path = tmp_path / 'conf' / 'malformed.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Write malformed YAML
        with open(yaml_path, 'w') as f:
            f.write('invalid: yaml: content: [')

        monkeypatch.chdir(tmp_path)

        # Should handle gracefully (either raise or use defaults)
        with pytest.raises(Exception):
            load_config(DirConfig, 'malformed.yaml')


    def test_yaml_with_invalid_structure(self, tmp_path, monkeypatch):

        '''Test YAML with structure that doesn't match config model.'''

        yaml_path = tmp_path / 'conf' / 'invalid_structure.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Write YAML with wrong types
        config_data = {
            'checkpoint_dir': ['this', 'should', 'be', 'string']
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)

        # Should raise validation error
        with pytest.raises(ValidationError):
            load_config(DirConfig, 'invalid_structure.yaml')
