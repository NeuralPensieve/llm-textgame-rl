from dataclasses import dataclass

@dataclass
class PPOConfig:
    # Model settings
    # model_name: str = "microsoft/DialoGPT-large"  # Smaller model for RTX 3060
    model_name: str = "openai-community/gpt2"
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 1e-5
    
    # PPO hyperparameters
    epsilon_clip: float = 0.2
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training settings
    num_envs: int = 16
    num_steps: int = 8  # Steps per rollout
    num_iterations: int = 100
    save_interval: int = 50
    log_interval: int = 10
    
    # LoRA settings
    lora_enabled: bool = True  # Enable LoRA
    lora_rank: int = 16  # Low-rank dimension for LoRA
    lora_dropout: float = 0.1  # Dropout for LoRA layers
    
    # Exploration
    epsilon: float = 0.5
    epsilon_decay: float = 0.95
    min_epsilon: float = 0.01