import wandb
import os
import argparse

from config import PPOConfig
from trainer import PPOTextWorldTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    """Main training function"""
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