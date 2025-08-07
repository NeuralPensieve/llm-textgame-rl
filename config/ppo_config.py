from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Model settings
    # model_name: str = "microsoft/DialoGPT-large"  # Smaller model for RTX 3060
    model_name: str = "openai-community/gpt2"
    max_length: int = 1024  # Maximum sequence length
    use_action_token_scoring = False  # False for "helpful" scoring, True for action token scoring
    

    # Environment
    num_envs: int = 16
    reuse_seed: bool = False
    env_seed: int = 142
    difficulty: str = 'easy'  # "easy", "medium", "hard"
    num_steps: int = 8  # Steps per rollout. 8 for easy, and 16 for medium

    # PPO hyperparameters
    batch_size: int = 1
    learning_rate: float = 1e-5
    epsilon_clip: float = 0.2
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    kl_loss_coef: float = 0.01
    accumulation_steps: int = 4  # Gradient accumulation steps
    normalize_advantage: bool = True

    # Training settings
    num_iterations: int = 200
    save_interval: int = 50
    log_interval: int = 10
    eval_interval: int = 5

    # LoRA settings
    lora_enabled: bool = False  # Enable LoRA
    lora_rank: int = 16  # Low-rank dimension for LoRA
    lora_dropout: float = 0.1  # Dropout for LoRA layers

    # Exploration
    epsilon: float = 0.5
    epsilon_decay: float = 0.95
    min_epsilon: float = 0.01
