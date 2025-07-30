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
    num_envs: int = 4
    num_steps: int = 8  # Steps per rollout
    num_iterations: int = 1000
    save_interval: int = 50
    log_interval: int = 10
    
    # Exploration
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05