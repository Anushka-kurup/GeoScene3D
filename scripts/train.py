"""
Training script for GeoCAR-Lite
"""

import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('src')

from models.geocar import GeoCAR
from data.temporal_dataset import TemporalDataset
from training.trainer import Trainer
from omegaconf import OmegaConf


def main():
    # Load config
    config = OmegaConf.load('configs/base_config.yaml')
    
    # Create model
    model = GeoCAR(config.model)
    
    # Create dataset
    train_dataset = TemporalDataset(
        config.data.train_path,
        config.data
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    
    # Create trainer
    trainer = Trainer(model, config.training)
    
    # Train
    trainer.train(train_loader)


if __name__ == '__main__':
    main()