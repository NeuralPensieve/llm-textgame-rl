from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Model settings
    # model_name: str = "microsoft/DialoGPT-small"
    # model_name: str = "openai-community/gpt2"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 750  # Maximum sequence length
    dynamic_config: bool = False
    debug_mode: bool = False
    
    # Easy settings
    # difficulty: str = 'easy'  # "easy", "medium", "hard"
    # num_steps: int = 8  # Steps per rollout. 8 for easy, and 16 for medium
    # repeatable: bool = False
    # num_iterations: int = 100
    
    # Medium settings
    difficulty: str = 'medium'  # "easy", "medium", "hard"
    num_steps: int = 32  # Steps per rollout. 8 for easy, and 16 for medium
    repeatable: bool = False
    num_iterations: int = 1000
    
    # Environment
    num_envs: int = 16
    reuse_seed: bool = False
    env_seed: int = 142
    step_penalty: float = 0.1
    history_len: int = 3

    # PPO hyperparameters
    batch_size: int = 2
    accumulation_steps: int = 32
    learning_rate: float = 1e-5
    learning_rate_value_head: float = 1e-4
    epsilon_clip: float = 0.2
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    # Training settings
    save_interval: int = 50
    log_interval: int = 10
    eval_interval: int = 5
    disable_value_function: bool = False  # Disable value function training
    use_fp16: bool = True

    # Loss settings
    value_loss_coef: float = 0.05
    entropy_coef: float = 0.01
    kl_coef: float = 0.1  # Start with 0.1, tune based on results

    # LoRA settings
    lora_enabled: bool = False  # Enable LoRA
    lora_rank: int = 16  # Low-rank dimension for LoRA
    lora_dropout: float = 0.1  # Dropout for LoRA layers
    lora_alpha: int = 64

    # Exploration
    epsilon: float = 0.0
    epsilon_decay: float = 0.95
    min_epsilon: float = 0.0
    temperature: float = 1.0
    min_temperature: float = 0.5
    temperature_decay: float = 0.995


    # Evaluation
    num_eval_episodes: int = 16
    num_sample_games: int = 5

    # KL Penalty settings
    use_kl_penalty: bool = False  # ONLY works with action_token
    reference_fp16: bool = False  # Use FP16 for reference model to save memory