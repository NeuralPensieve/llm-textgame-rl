import wandb

from config.ppo_config import PPOConfig
from trainers import PPOTextWorldTrainer

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    """Main training function"""
    # Configuration
    config = PPOConfig()
    
    # Create trainer and start training
    trainer = PPOTextWorldTrainer(config)
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    main()