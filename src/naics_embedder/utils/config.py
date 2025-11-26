# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


# -------------------------------------------------------------------------------------------------
# Generic Config Loader
# -------------------------------------------------------------------------------------------------

def load_config(config_class: Type[T], yaml_path: Union[str, Path]) -> T:

    '''
    Generic configuration loader for any Pydantic model.
    
    Args:
        config_class: The Pydantic model class to instantiate
        yaml_path: Path to YAML config file (absolute, relative, or under conf/)
        
    Returns:
        Instance of config_class with values from YAML
        
    Example:
        cfg = load_config(DownloadConfig, 'data/download.yaml')
    '''
    
    path = Path(yaml_path)
    search_paths: List[Path] = []
    
    if path.is_absolute():
        search_paths.append(path)
    else:
        normalized = path.as_posix()
        search_paths.append(path)

        if not normalized.startswith('conf/'):
            search_paths.append(Path('conf') / path)
    
    yaml_file: Optional[Path] = None
    for candidate in search_paths:
        candidate_path = candidate.expanduser()
        if candidate_path.exists():
            yaml_file = candidate
            break
    
    if yaml_file is None:
        logger.warning(f'Config file not found: {path}, using defaults')
        return config_class()
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}
    
    logger.info(f'Loaded config from {yaml_path}')
    
    return config_class(**data)


# -------------------------------------------------------------------------------------------------
# Directory Configuration
# -------------------------------------------------------------------------------------------------

class DirConfig(BaseModel):

    '''File system directory configuration.'''
        
    checkpoint_dir: str = Field(
        default='./checkpoints',
        description='Directory for model checkpoints'
    )
    conf_dir: str = Field(
        default='./conf',
        description='Directory for config files'
    )
    data_dir: str = Field(
        default='./data',
        description='Directory containing data files'
    )
    docs_dir: str = Field(
        default='./docs',
        description='Directory containing data files'
    )
    log_dir: str = Field(
        default='./logs',
        description='Directory for logs'
    )
    output_dir: str = Field(
        default='./outputs',
        description='Directory for training outputs'
    )


# -------------------------------------------------------------------------------------------------
# Data Generation Configuration
# -------------------------------------------------------------------------------------------------

class DownloadConfig(BaseModel):

    '''Configuration for downloading and preprocessing NAICS data.'''
    
    output_parquet: str = Field(
        default='./data/naics_descriptions.parquet',
        description='Output path for processed descriptions'
    )
    
    # URLs for data sources
    url_codes: str = Field(
        default='https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx',
        description='URL for NAICS codes Excel file'
    )
    url_index: str = Field(
        default='https://www.census.gov/naics/2022NAICS/2022_NAICS_Index_File.xlsx',
        description='URL for NAICS index file'
    )
    url_descriptions: str = Field(
        default='https://www.census.gov/naics/2022NAICS/2022_NAICS_Descriptions.xlsx',
        description='URL for NAICS descriptions'
    )
    url_exclusions: str = Field(
        default='https://www.census.gov/naics/2022NAICS/2022_NAICS_Cross_References.xlsx',
        description='URL for NAICS cross references'
    )
    
    # Sheet names
    sheet_codes: str = Field(
        default='tbl_2022_title_description_coun',
        description='Sheet name for codes'
    )
    sheet_index: str = Field(
        default='2022NAICS',
        description='Sheet name for index'
    )
    sheet_descriptions: str = Field(
        default='2022_NAICS_Descriptions',
        description='Sheet name for descriptions'
    )
    sheet_exclusions: str = Field(
        default='2022_NAICS_Cross_References',
        description='Sheet name for exclusions'
    )
    
    # Schema definitions (stored as JSON-serializable dicts)
    schema_codes: Dict[str, str] = Field(
        default={
            'Seq. No.': 'UInt32',
            '2022 NAICS US   Code': 'Utf8',
            '2022 NAICS US Title': 'Utf8',
        },
        description='Schema for codes sheet'
    )
    schema_index: Dict[str, str] = Field(
        default={'NAICS22': 'Utf8', 'INDEX ITEM DESCRIPTION': 'Utf8'},
        description='Schema for index sheet'
    )
    schema_descriptions: Dict[str, str] = Field(
        default={'Code': 'Utf8', 'Description': 'Utf8'},
        description='Schema for descriptions sheet'
    )
    schema_exclusions: Dict[str, str] = Field(
        default={'Code': 'Utf8', 'Cross-Reference': 'Utf8'},
        description='Schema for exclusions sheet'
    )
    
    # Column renames
    rename_codes: Dict[str, str] = Field(
        default={
            'Seq. No.': 'index',
            '2022 NAICS US   Code': 'code',
            '2022 NAICS US Title': 'title',
        },
        description='Column renames for codes'
    )
    rename_index: Dict[str, str] = Field(
        default={'NAICS22': 'code', 'INDEX ITEM DESCRIPTION': 'examples'},
        description='Column renames for index'
    )
    rename_descriptions: Dict[str, str] = Field(
        default={'Code': 'code', 'Description': 'description'},
        description='Column renames for descriptions'
    )
    rename_exclusions: Dict[str, str] = Field(
        default={'Code': 'code', 'Cross-Reference': 'excluded'},
        description='Column renames for exclusions'
    )
    
    @field_validator('output_parquet')
    @classmethod
    def validate_output_parquet(cls, value: str) -> str:
        path = Path(value)
        if path.suffix.lower() != '.parquet':
            raise ValueError('output_parquet must point to a .parquet file')
        return value
    

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DownloadConfig':
        
        '''Load configuration from YAML file.'''

        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            logger.warning(f'Config file not found: {yaml_path}, using defaults')
            return cls()
        
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)


class RelationsConfig(BaseModel):
    
    '''Configuration for computing pairwise relations.'''
    
    input_parquet: str = Field(
        default='./data/naics_descriptions.parquet',
        description='Input descriptions parquet file'
    )
    output_parquet: str = Field(
        default='./data/naics_relations.parquet',
        description='Output relations parquet file'
    )
    relation_matrix_parquet: str = Field(
        default='./data/naics_relation_matrix.parquet',
        description='Output relation matrix parquet file'
    )
    
    relation_id: Dict[str, int] = Field(
        default={
            'child': 1,
            'sibling': 2,
            'grandchild': 3,
            'great-grandchild': 4,
            'nephew/niece': 5,
            'great-great-grandchild': 6,
            'cousin': 7,
            'grand-nephew/niece': 8,
            'grand-grand-nephew/niece': 9,
            'cousin_1_times_removed': 10,
            'second_cousin': 11,
            'cousin_2_times_removed': 12,
            'second_cousin_1_times_removed': 13,
            'third_cousin': 14,
        },
        description='Mapping of relation names to IDs'
    )
    

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RelationsConfig':
    
        '''Load configuration from YAML file.'''
    
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            logger.warning(f'Config file not found: {yaml_path}, using defaults')
            return cls()
        
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)


class DistancesConfig(BaseModel):

    '''Configuration for computing pairwise distances.'''
    
    input_parquet: str = Field(
        default='./data/naics_descriptions.parquet',
        description='Input descriptions parquet file'
    )
    distances_parquet: str = Field(
        default='./data/naics_distances.parquet',
        description='Output distances parquet file'
    )
    distance_matrix_parquet: str = Field(
        default='./data/naics_distance_matrix.parquet',
        description='Output distance matrix parquet file'
    )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DistancesConfig':
        
        '''Load configuration from YAML file.'''
        
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            logger.warning(f'Config file not found: {yaml_path}, using defaults')
            return cls()
        
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)


class TripletsConfig(BaseModel):

    '''Configuration for generating training triplets.'''
    
    descriptions_parquet: str = Field(
        default='./data/naics_descriptions.parquet',
        description='Input descriptions parquet file'
    )
    distances_parquet: str = Field(
        default='./data/naics_distances.parquet',
        description='Input distances parquet file'
    )
    relations_parquet: str = Field(
        default='./data/naics_relations.parquet',
        description='Input relations parquet file'
    )
    output_parquet: str = Field(
        default='./data/naics_training_pairs',
        description='Output directory for training pairs'
    )
    
    # Anchor parameters
    anchor_level: Optional[List[int]] = Field(
        default=None,
        description='Filter anchor codes by hierarchy level'
    )
    relation_margin: Optional[List[int]] = Field(
        default=None,
        description='Filter by relation margin'
    )
    distance_margin: Optional[List[float]] = Field(
        default=None,
        description='Filter by distance margin'
    )
    
    # Margin parameters
    positive_level: Optional[List[int]] = Field(
        default=None,
        description='Filter positive codes by hierarchy level'
    )
    positive_relation: Optional[List[int]] = Field(
        default=None,
        description='Filter positive pairs by relation'
    )
    positive_distance: Optional[List[float]] = Field(
        default=None,
        description='Filter positive pairs by distance'
    )
    n_positives: int = Field(
        default=2125,
        gt=0,
        le=2125,
        description='Maximum number of positives per anchor'
    )
    
    # Margin parameters
    negative_level: Optional[List[int]] = Field(
        default=None,
        description='Filter negative codes by hierarchy level'
    )
    negative_relation: Optional[List[int]] = Field(
        default=None,
        description='Filter negative pairs by relation'
    )
    negative_distance: Optional[List[int]] = Field(
        default=None,
        description='Filter negative pairs by distance'
    )
    n_negatives: int = Field(
        default=2125,
        gt=0,
        le=2125,
        description='Maximum number of negatives per positive'
    )
    

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TripletsConfig':

        '''Load configuration from YAML file.'''
        
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            logger.warning(f'Config file not found: {yaml_path}, using defaults')
            return cls()
        
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)


class DataConfig(BaseModel):

    '''Data generation configuration.'''
    
    download: DownloadConfig = Field(
        default_factory=DownloadConfig,
        description='Download configuration'
    )
    relations: RelationsConfig = Field(
        default_factory=RelationsConfig,
        description='Relations configuration'
    )    
    distances: DistancesConfig = Field(
        default_factory=DistancesConfig,
        description='Distances configuration'
    )
    triplets: TripletsConfig = Field(
        default_factory=TripletsConfig,
        description='Triplets configuration'
    )


# -------------------------------------------------------------------------------------------------
# Data Loader Configuration
# -------------------------------------------------------------------------------------------------

class TokenizationConfig(BaseModel):

    '''Configuration for tokenization caching.'''
    
    descriptions_parquet: str = Field(
        default='./data/naics_descriptions.parquet',
        description='Path to descriptions parquet file'
    )
    tokenizer_name: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        description='HuggingFace tokenizer name'
    )
    max_length: Optional[int] = Field(
        default=None,
        description='Maximum sequence length (None = use model default)'
    )
    output_path: str = Field(
        default='./data/token_cache/token_cache.pt',
        description='Path to save tokenization cache'
    )
    

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TokenizationConfig':

        '''Load configuration from YAML file.'''
        
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            logger.warning(f'Config file not found: {yaml_path}, using defaults')
            return cls()
        
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)


class StreamingConfig(BaseModel):

    '''Configuration for streaming dataset.'''
    
    descriptions_parquet: str = Field(
        default='./data/naics_descriptions.parquet',
        description='Path to descriptions parquet file'
    )
    distances_parquet: str = Field(
        default='./data/naics_distances.parquet',
        description='Path to distances parquet file'
    )
    distance_matrix_parquet: str = Field(
        default='./data/naics_distance_matrix.parquet',
        description='Path to distance matrix parquet file'
    )
    relations_parquet: str = Field(
        default='./data/naics_relations.parquet',
        description='Path to relations parquet file'
    )
    triplets_parquet: str = Field(
        default='./data/naics_training_pairs',
        description='Path to training pairs directory'
    )
    tokenizer_name: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        description='HuggingFace tokenizer name'
    )
    max_length: int = Field(
        default=512,
        description='Maximum sequence length for tokenization'
    )
    seed: int = Field(
        default=42,
        ge=0,
        description='Random seed for sampling'
    )
    
    # Anchor parameters
    anchor_level: Optional[List[int]] = Field(
        default=None,
        description='Filter anchor codes by hierarchy level'
    )
    relation_margin: Optional[List[int]] = Field(
        default=None,
        description='Filter by relation margin'
    )
    distance_margin: Optional[List[int]] = Field(
        default=None,
        description='Filter by distance margin'
    )
    
    # Margin parameters
    positive_level: Optional[List[int]] = Field(
        default=None,
        description='Filter positive codes by hierarchy level'
    )
    positive_relation: Optional[List[int]] = Field(
        default=None,
        description='Filter positive pairs by relation'
    )
    positive_distance: Optional[List[int]] = Field(
        default=None,
        description='Filter positive pairs by distance'
    )
    
    # Margin parameters
    negative_level: Optional[List[int]] = Field(
        default=None,
        description='Filter negative codes by hierarchy level'
    )
    negative_relation: Optional[List[int]] = Field(
        default=None,
        description='Filter negative pairs by relation'
    )
    negative_distance: Optional[List[int]] = Field(
        default=None,
        description='Filter negative pairs by distance'
    )
    
    # Sampling parameters
    n_positives: int = Field(
        default=2125,
        gt=0,
        description='Maximum number of positives per anchor'
    )
    n_negatives: int = Field(
        default=2125,
        gt=0,
        description='Maximum number of negatives per positive'
    )
    
    # Phase 1 sampling parameters
    use_phase1_sampling: bool = Field(
        default=False,
        description=(
            'Use Phase 1 tree-distance based sampling '
            '(inverse weighting, sibling masking, exclusion mining)'
        )
    )
    phase1_alpha: float = Field(
        default=1.5,
        gt=0.0,
        description='Exponent for inverse tree distance weighting: P(n) ∝ 1 / D_tree(a, n)^α'
    )
    phase1_exclusion_weight: float = Field(
        default=100.0,
        gt=0.0,
        description='High constant weight for excluded codes in Phase 1 sampling'
    )


class DataLoaderConfig(BaseModel):

    '''Data loading and preprocessing configuration.'''
    
    tokenization: TokenizationConfig = Field(
        default_factory=TokenizationConfig,
        description='Tokenization configuration'
    )
    streaming: StreamingConfig = Field(
        default_factory=StreamingConfig,
        description='Streaming configuration'
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        le=512,
        description='Training batch size'
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        le=32,
        description='Number of data loading workers'
    )
    val_split: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description='Validation split fraction'
    )
    

    @field_validator('batch_size')
    @classmethod
    def warn_large_batch(cls, v: int) -> int:
    
        '''Warn about potentially problematic batch sizes.'''
    
        if v > 128:
            logger.warning(f'Large batch_size={v} may cause OOM errors')
        return v


# -------------------------------------------------------------------------------------------------
# Model Configuration
# -------------------------------------------------------------------------------------------------

class LoRAConfig(BaseModel):

    '''LoRA (Low-Rank Adaptation) configuration.'''
    
    r: int = Field(
        default=8,
        gt=0,
        le=64,
        description='LoRA rank (lower = fewer parameters)'
    )
    alpha: int = Field(
        default=16,
        gt=0,
        description='LoRA scaling factor'
    )
    dropout: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description='LoRA dropout rate'
    )


class MoEConfig(BaseModel):

    '''Mixture of Experts configuration.'''
        
    num_experts: int = Field(
        default=4,
        gt=0,
        le=16,
        description='Number of expert networks'
    )
    top_k: int = Field(
        default=2,
        gt=0,
        description='Number of experts to activate per input'
    )
    hidden_dim: int = Field(
        default=1024,
        gt=0,
        description='Hidden dimension for expert networks'
    )
    load_balancing_coef: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description='Load balancing loss coefficient'
    )
    

    @model_validator(mode='after')
    def validate_top_k_vs_experts(self) -> 'MoEConfig':
      
        '''Ensure top_k doesn't exceed num_experts.'''
      
        if self.top_k > self.num_experts:
            raise ValueError(
                f'top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})'
            )
        return self


class ModelConfig(BaseModel):
    
    '''Model architecture configuration.'''
    
    base_model_name: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        description='HuggingFace base model name'
    )
    lora: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description='LoRA configuration'
    )
    moe: MoEConfig = Field(
        default_factory=MoEConfig,
        description='Mixture of Experts configuration'
    )
    eval_sample_size: int = Field(
        default=500,
        gt=0,
        le=2125,
        description='Number of codes to sample for evaluation'
    )
    eval_every_n_epochs: int = Field(
        default=1,
        gt=0,
        description='Run evaluation every N epochs'
    )


# -------------------------------------------------------------------------------------------------
# Loss Configuration
# -------------------------------------------------------------------------------------------------

class LossConfig(BaseModel):
  
    '''Loss function configuration.'''
    
    temperature: float = Field(
        default=0.07,
        gt=0,
        le=1,
        description='Temperature for contrastive loss'
    )
    curvature: float = Field(
        default=1.0,
        gt=0,
        description='Curvature for hyperbolic space'
    )
    hierarchy_weight: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description='Weight for hierarchy preservation loss component (0.0 to disable)'
    )
    rank_order_weight: float = Field(
        default=0.15,
        ge=0,
        le=1.0,
        description=(
            'Weight for rank order preservation loss '
            '(Spearman correlation optimization, 0.0 to disable)'
        )
    )
    radius_reg_weight: float = Field(
        default=0.01,
        ge=0,
        le=1.0,
        description=(
            'Weight for radius regularization to prevent hyperbolic radius '
            'instability (0.0 to disable)'
        )
    )


# -------------------------------------------------------------------------------------------------
# Training Configuration
# -------------------------------------------------------------------------------------------------

class TrainerConfig(BaseModel):
   
    '''PyTorch Lightning Trainer configuration.'''
    
    max_epochs: int = Field(
        default=10,
        gt=0,
        le=1000,
        description='Maximum number of training epochs'
    )
    accelerator: str = Field(
        default='auto',
        description='Training accelerator (auto, gpu, cpu, mps)'
    )
    devices: int = Field(
        default=1,
        gt=0,
        description='Number of devices to use'
    )
    precision: str = Field(
        default='16-mixed',
        description='Training precision (32, 16-mixed, bf16-mixed)'
    )
    gradient_clip_val: float = Field(
        default=1.0,
        gt=0,
        description='Gradient clipping value'
    )
    accumulate_grad_batches: int = Field(
        default=1,
        gt=0,
        description='Number of batches for gradient accumulation'
    )
    log_every_n_steps: int = Field(
        default=10,
        gt=0,
        description='Log metrics every N steps'
    )
    val_check_interval: float = Field(
        default=1.0,
        gt=0,
        description='Run validation every N epochs (or fraction)'
    )
    

    @field_validator('accelerator')
    @classmethod
    def validate_accelerator(cls, v: str) -> str:
    
        '''Validate accelerator choice.'''
    
        valid = ['auto', 'gpu', 'cpu', 'mps', 'cuda']
        if v not in valid:
            raise ValueError(f'accelerator must be one of {valid}')
        return v
    

    @field_validator('precision')
    @classmethod
    def validate_precision(cls, v: str) -> str:
     
        '''Validate precision choice.'''
     
        valid = ['32', '16', '16-mixed', 'bf16', 'bf16-mixed']
        if v not in valid:
            raise ValueError(f'precision must be one of {valid}')
        return v


class CurriculumConfig(BaseModel):

    '''Structure-Aware Dynamic Curriculum (SADC) scheduler configuration.'''

    phase1_end: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description='End of Phase 1 (Structural Initialization) as fraction of max epochs'
    )
    phase2_end: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description='End of Phase 2 (Geometric Refinement) as fraction of max epochs'
    )
    phase3_end: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description='End of Phase 3 (False Negative Mitigation) as fraction of max epochs'
    )
    tree_distance_alpha: float = Field(
        default=1.5,
        gt=0,
        description='Exponent for inverse tree-distance weighting of negatives'
    )
    sibling_distance_threshold: float = Field(
        default=2.0,
        ge=0,
        description='Distance threshold for sibling masking in Phase 1'
    )
    fn_curriculum_start_epoch: int = Field(
        default=10,
        ge=0,
        description='Epoch to begin clustering-based false-negative elimination'
    )
    fn_cluster_every_n_epochs: int = Field(
        default=5,
        gt=0,
        description='Frequency (in epochs) for refreshing clustering in Phase 3'
    )
    fn_num_clusters: int = Field(
        default=500,
        gt=0,
        description='Number of clusters used in false-negative elimination'
    )

    @model_validator(mode='after')
    def validate_phase_boundaries(self) -> 'CurriculumConfig':
        '''Ensure curriculum phases progress monotonically.''' 

        if not (self.phase1_end <= self.phase2_end <= self.phase3_end):
            raise ValueError(
                'Curriculum phases must satisfy '
                'phase1_end <= phase2_end <= phase3_end'
            )
        return self


class TrainingConfig(BaseModel):
    
    '''Optimizer and training configuration.'''
    
    learning_rate: float = Field(
        default=2e-4,
        gt=0,
        lt=1,
        description='Learning rate for optimizer'
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        lt=1,
        description='Weight decay (L2 regularization)'
    )
    warmup_steps: int = Field(
        default=500,
        ge=0,
        description='Number of warmup steps'
    )
    use_warmup_cosine: bool = Field(
        default=False,
        description='Use warmup + cosine decay scheduler instead of ReduceLROnPlateau. '
                    'Beneficial for large training jobs with many epochs.'
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description='PyTorch Lightning Trainer config'
    )




# -------------------------------------------------------------------------------------------------
# Training Configuration
# -------------------------------------------------------------------------------------------------


class GraphConfig(BaseModel):
    '''Base configuration for HGCN training.'''
    
    encodings_parquet: str = Field(
        default='./output/hyperbolic_projection/encodings.parquet',
        description='Path to input hyperbolic embeddings parquet file'
    )
    relations_parquet: str = Field(
        default='./data/naics_relations.parquet',
        description='Path to relations parquet file'
    )
    training_pairs_path: str = Field(
        default='./data/naics_training_pairs.parquet',
        description='Path to training pairs directory'
    )
    
    output_dir: str = Field(
        default='./output/hgcn/',
        description='Output directory for checkpoints and logs'
    )
    output_parquet: str = Field(
        default='./output/hgcn/encodings.parquet',
        description='Output path for final embeddings'
    )
    
    tangent_dim: int = Field(
        default=31,
        gt=0,
        description='Spatial dimensions in tangent space'
    )
    n_hgcn_layers: int = Field(
        default=2,
        gt=0,
        description='Number of HGCN layers'
    )
    dropout: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description='Dropout rate'
    )
    learnable_curvature: bool = Field(
        default=True,
        description='Whether curvature is learnable'
    )
    learnable_loss_weights: bool = Field(
        default=True,
        description='Whether loss weights are learnable'
    )
    
    # Training parameters
    num_epochs: int = Field(
        default=8,
        gt=0,
        description='Number of training epochs'
    )
    epoch_every: int = Field(
        default=1,
        gt=0,
        description='Log every N epochs'
    )
    batch_size: int = Field(
        default=64,
        gt=0,
        description='Batch size'
    )
    lr: float = Field(
        default=0.00075,
        gt=0,
        description='Learning rate'
    )
    warmup_ratio: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description='Fraction of epochs for warmup'
    )
    temperature_start: float = Field(
        default=0.10,
        gt=0,
        description='Starting temperature for loss scaling'
    )
    temperature_end: float = Field(
        default=0.08,
        gt=0,
        description='Ending temperature for loss scaling'
    )
    weight_decay: float = Field(
        default=1e-5,
        ge=0,
        description='Weight decay'
    )
    gradient_clip_norm: float = Field(
        default=1.0,
        gt=0,
        description='Gradient clipping norm'
    )
    
    # Loss parameters
    triplet_margin: float = Field(
        default=1.0,
        gt=0,
        description='Triplet loss margin'
    )
    w_triplet: float = Field(
        default=1.0,
        ge=0,
        description='Triplet loss weight'
    )
    w_per_level: float = Field(
        default=0.5,
        ge=0,
        description='Level loss weight'
    )
    
    # Data sampling parameters (legacy, not used in weighted sampling)
    k_total: int = Field(
        default=48,
        gt=0,
        description='Total number of negatives to sample'
    )
    pct_excluded: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description='Percentage of excluded negatives'
    )
    pct_hard: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description='Percentage of hard negatives'
    )
    pct_medium: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description='Percentage of medium negatives'
    )
    pct_easy: float = Field(
        default=0.55,
        ge=0,
        le=1,
        description='Percentage of easy negatives'
    )
    pct_unrelated: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description='Percentage of unrelated negatives'
    )
    
    n_positive_samples: int = Field(
        default=2048,
        gt=0,
        description='Number of positive samples'
    )
    allowed_relations: Optional[List[str]] = Field(
        default=None,
        description='Allowed relations for filtering'
    )
    min_code_level: Optional[int] = Field(
        default=None,
        ge=2,
        le=6,
        description='Minimum code level'
    )
    max_code_level: Optional[int] = Field(
        default=None,
        ge=2,
        le=6,
        description='Maximum code level'
    )
    
    shuffle: bool = Field(
        default=True,
        description='Shuffle data'
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description='Number of data loader workers'
    )
    pin_memory: bool = Field(
        default=True,
        description='Pin memory for data loader'
    )
    
    device: str = Field(
        default='auto',
        description='Device string'
    )
    seed: int = Field(
        default=42,
        ge=0,
        description='Random seed'
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'GraphConfig':
        '''Load GraphConfig from YAML file.'''

        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f'Config file not found: {yaml_path}')

        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        logger.info(f'Loaded graph config from {yaml_path}')

        return cls(**data)


# -------------------------------------------------------------------------------------------------
# Main Configuration
# -------------------------------------------------------------------------------------------------

class Config(BaseModel):

    '''Main configuration for NAICS training.'''
    
    experiment_name: str = Field(
        default='default',
        description='Experiment name for logging and checkpoints'
    )
    seed: int = Field(
        default=42,
        ge=0,
        description='Random seed for reproducibility'
    )
    curriculum: CurriculumConfig = Field(
        default_factory=CurriculumConfig,
        description='Dynamic SADC curriculum scheduler configuration'
    )
    dirs: DirConfig = Field(
        default_factory=DirConfig,
        description='File system paths'
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description='Data generation configuration'
    )
    data_loader: DataLoaderConfig = Field(
        default_factory=DataLoaderConfig,
        description='Data loading configuration'
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description='Model architecture configuration'
    )
    loss: LossConfig = Field(
        default_factory=LossConfig,
        description='Loss function configuration'
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description='Training configuration'
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        '''Load configuration from YAML file.'''

        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f'Config file not found: {yaml_path}')

        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        logger.info(f'  • Loaded config from {yaml_path}\n')

        return cls(**data)
    

    def override(self, overrides: Dict[str, Any]) -> 'Config':
   
        '''Apply overrides using dot notation.'''
   
        data = self.model_dump()
        
        for key, value in overrides.items():
            parts = key.split('.')
            current = data
            
            # Navigate to nested dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set value
            current[parts[-1]] = value
        
        logger.info(f'Applied {len(overrides)} override(s)')
        return Config(**data)
    

    def to_dict(self) -> Dict[str, Any]:
    
        '''Convert config to dictionary.'''
    
        return self.model_dump()
    

    def to_yaml(self, path: str) -> None:
    
        '''Save config to YAML file.'''
    
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f'Saved config to {path}')
    

    class ConfigDict:
    
        '''Pydantic v2 configuration.'''
    
        validate_assignment = True
        extra = 'forbid'
        str_strip_whitespace = True


# -------------------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------------------

def parse_override_value(value: str) -> Any:

    '''Parse override value from string to appropriate type.'''

    try:
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        if '.' in value or 'e' in value.lower():
            return float(value)

        try:
            return int(value)
        except ValueError:
            pass

        import ast
        return ast.literal_eval(value)

    except (ValueError, SyntaxError):
        return value
