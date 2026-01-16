"""
Training loop for GeoCAR
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer for GeoCAR model
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = None
        
        # Setup logging
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=config
            )
        
        # Device
        self.device = next(model.parameters()).device
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
    
    def _setup_optimizer(self):
        """
        Setup optimizer - only train novel modules
        """
        # Get trainable parameters
        trainable_params = []
        
        # Change fusion
        trainable_params.extend(self.model.change_fusion.parameters())
        
        # Verifier
        trainable_params.extend(self.model.verifier.parameters())
        
        # Process tokens
        trainable_params.append(self.model.process_tokens)
        trainable_params.extend(self.model.process_attention.parameters())
        
        # LoRA parameters in VLM
        for name, param in self.model.base_vlm.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def train(self, train_loader, val_loader=None):
        """
        Main training loop
        """
        from .losses import GeoCarLoss, compute_gt_plausibility
        
        criterion = GeoCarLoss(self.config)
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self._train_epoch(train_loader, criterion)
            
            # Validate
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)
            
            # Log
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **(val_metrics if val_loader else {})
                })
    
    def _train_epoch(self, train_loader, criterion):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_lm_loss = 0
        total_ver_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            img_t1 = batch['img_t1'].to(self.device)
            img_t2 = batch['img_t2'].to(self.device)
            questions = batch['questions']
            descriptions = batch['descriptions']
            
            # Forward
            outputs = self.model(
                img_t1, img_t2, questions,
                labels=None  # TODO: tokenize descriptions as labels
            )
            
            # Compute ground truth plausibility
            gt_plausibility = compute_gt_plausibility(
                descriptions,
                batch['constraints']
            ).to(self.device)
            
            targets = {
                'gt_plausibility': gt_plausibility,
                'process_stages': batch['process_stages']
            }
            
            # Compute loss
            loss, loss_dict = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Track
            total_loss += loss.item()
            total_lm_loss += loss_dict['lm_loss'].item()
            total_ver_loss += loss_dict['verification_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lm': loss_dict['lm_loss'].item(),
                'ver': loss_dict['verification_loss'].item()
            })
            
            # Log
            if self.global_step % self.config.log_every_n_steps == 0:
                if self.config.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lm_loss': loss_dict['lm_loss'].item(),
                        'train/verification_loss': loss_dict['verification_loss'].item(),
                        'train/change_loss': loss_dict['change_loss'].item(),
                        'lr': self.optimizer.param_groups[0]['lr']
                    }, step=self.global_step)
            
            self.global_step += 1
        
        return {
            'train/loss': total_loss / len(train_loader),
            'train/lm_loss': total_lm_loss / len(train_loader),
            'train/ver_loss': total_ver_loss / len(train_loader)
        }
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                img_t1 = batch['img_t1'].to(self.device)
                img_t2 = batch['img_t2'].to(self.device)
                questions = batch['questions']
                
                # Forward
                outputs = self.model.generate(img_t1, img_t2, questions)
                
                # TODO: Compute validation metrics
        
        return {
            'val/loss': total_loss / len(val_loader)
        }
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f'checkpoint_epoch{epoch+1}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step
        }, checkpoint_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")