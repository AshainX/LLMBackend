import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SimpleLLM, ModelConfig
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
from tqdm import tqdm

class TrainerConfig:
    def __init__(self,
                 learning_rate=3e-4,
                 min_learning_rate=1e-5,
                 warmup_steps=1000,
                 weight_decay=0.01,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 adam_epsilon=1e-8,
                 gradient_clip_norm=1.0,
                 gradient_accumulation_steps=1,
                 mixed_precision=True,
                 log_interval=100,
                 eval_interval=1000,
                 save_interval=1000,
                 checkpoint_dir='checkpoints',
                 use_wandb=False):
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb

class Trainer:
    def __init__(self, 
                 model: SimpleLLM,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 config: TrainerConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainerConfig()
        self.device = device
        
        # Initialize optimizer
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm'])],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm'])],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon
        )
        
        # Initialize learning rate scheduler
        self.lr_scheduler = self._create_scheduler()
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # Create checkpoint directory
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
            
        # Initialize wandb
        if self.config.use_wandb:
            wandb.init(project="llm-training")
            
    def _create_scheduler(self):
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader),
            eta_min=self.config.min_learning_rate
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_steps]
        )
        
    def save_checkpoint(self, step, loss):
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
        }
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint-{step}.pt')
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step']
        
    def train_step(self, batch, step):
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        
        # Mixed precision training
        with autocast(enabled=self.config.mixed_precision):
            output = self.model(data)
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            loss = F.cross_entropy(output, target)
            loss = loss / self.config.gradient_accumulation_steps
            
        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Update weights if gradient accumulation is complete
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            
            # Optimizer step
            if self.config.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            
        return loss.item() * self.config.gradient_accumulation_steps
        
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                with autocast(enabled=self.config.mixed_precision):
                    output = self.model(data)
                    output = output.view(-1, output.size(-1))
                    target = target.view(-1)
                    loss = F.cross_entropy(output, target)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
        
    def train(self, total_steps):
        self.model.train()
        step = 0
        train_iterator = iter(self.train_loader)
        
        with tqdm(total=total_steps, desc="Training") as pbar:
            while step < total_steps:
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(self.train_loader)
                    batch = next(train_iterator)
                    
                loss = self.train_step(batch, step)
                
                if (step + 1) % self.config.log_interval == 0:
                    lr = self.lr_scheduler.get_last_lr()[0]
                    metrics = {
                        'loss': loss,
                        'learning_rate': lr,
                        'step': step
                    }
                    
                    if self.config.use_wandb:
                        wandb.log(metrics)
                    pbar.set_postfix(metrics)
                    
                if (step + 1) % self.config.eval_interval == 0 and self.val_loader:
                    val_loss = self.validate()
                    if self.config.use_wandb:
                        wandb.log({'val_loss': val_loss})
                    self.model.train()
                    
                if (step + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(step + 1, loss)
                    
                step += 1
                pbar.update(1) 