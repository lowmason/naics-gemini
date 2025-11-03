import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from naics_gemini.data_loader.datamodule import NAICSDataModule
from naics_gemini.model.naics_model import NAICSContrastiveModel
from naics_gemini.utils.console import configure_logging

logger = logging.getLogger(__name__)


def train(config_name: str = 'config', overrides: list = None):
    
    configure_logging()
    
    GlobalHydra.instance().clear()
    initialize(config_path='../conf', job_name='naics_train')
    
    cfg = compose(config_name=config_name, overrides=overrides or [])
    
    logger.info('Training Configuration:')
    logger.info(OmegaConf.to_yaml(cfg))
    
    pl.seed_everything(cfg.seed)
    
    datamodule = NAICSDataModule(
        descriptions_path=cfg.data.descriptions_path,
        triplets_path=cfg.data.triplets_path,
        tokenizer_name=cfg.data.tokenizer_name,
        curriculum_config={
            'positive_levels': cfg.curriculum.positive_levels,
            'positive_distance_min': cfg.curriculum.positive_distance_min,
            'positive_distance_max': cfg.curriculum.positive_distance_max,
            'max_positives': cfg.curriculum.max_positives,
            'difficulty_buckets': cfg.curriculum.difficulty_buckets,
            'bucket_percentages': cfg.curriculum.bucket_percentages,
            'k_negatives': cfg.curriculum.k_negatives
        },
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        seed=cfg.seed
    )
    
    model = NAICSContrastiveModel(
        base_model_name=cfg.model.base_model_name,
        lora_r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        use_moe=cfg.model.moe.enabled,
        num_experts=cfg.model.moe.num_experts,
        top_k=cfg.model.moe.top_k,
        moe_hidden_dim=cfg.model.moe.hidden_dim,
        temperature=cfg.loss.temperature,
        curvature=cfg.loss.curvature,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        load_balancing_coef=cfg.model.moe.load_balancing_coef
    )
    
    checkpoint_dir = Path(cfg.paths.checkpoint_dir) / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='naics-{epoch:02d}-{val/contrastive_loss:.4f}',
        monitor='val/contrastive_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val/contrastive_loss',
        patience=5,
        mode='min'
    )
    
    tb_logger = TensorBoardLogger(
        save_dir=cfg.paths.output_dir,
        name=cfg.experiment_name
    )
    
    trainer = pl.Trainer(
        **cfg.training.trainer,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        default_root_dir=cfg.paths.output_dir
    )
    
    logger.info('Starting training...')
    trainer.fit(model, datamodule)
    
    logger.info('Training complete!')
    logger.info(f'Best checkpoint: {checkpoint_callback.best_model_path}')
    
    return model, datamodule, trainer


if __name__ == '__main__':
    train()
