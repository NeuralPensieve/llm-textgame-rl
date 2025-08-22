import wandb
import os
import argparse
import random
import numpy as np
import torch

from config import PPOConfig
from trainer import PPOTextWorldTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed_and_determinism(seed=42):
    """
    Sets the seed for all random number generators and enforces
    deterministic behavior in PyTorch operations.
    """
    # Set seeds for libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU

    print(f"✅ Seeds set to {seed}")

def main():
    """Main training function"""
    # set_seed_and_determinism(42)

    parser = argparse.ArgumentParser(description="PPO TextWorld Training")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    args = parser.parse_args()
    
    # Configuration
    config = PPOConfig()

    

    print("Using standard PPO training")
    trainer = PPOTextWorldTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
            print(f"Resumed from iteration {trainer.iteration}")
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
            exit(1)
    
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()