import wandb

from config.ppo_config import PPOConfig
from trainers import PPOTextWorldTrainer, PPOLoRATextWorldTrainer

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    """Main training function"""
    # Configuration
    config = PPOConfig()

    if config.lora_enabled:
        print("Using LoRA adaptation for PPO training")
        trainer = PPOLoRATextWorldTrainer(config)
    else:
        print("Using standard PPO training")
        trainer = PPOTextWorldTrainer(config)
    
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    main()