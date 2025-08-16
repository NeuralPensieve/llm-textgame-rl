from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"  # Smaller model for RTX 3060
    # model_name: str = "openai-community/gpt2"
    # model_name: str = "google/gemma-3-270m-it"
    # model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 512  # Maximum sequence length
    scoring_method: str = "action_token"  # Options: "helpful" or "action_token"
    dynamic_config: bool = False
    debug_mode: bool = False
    

    # Environment
    num_envs: int = 16
    reuse_seed: bool = False
    env_seed: int = 142
    difficulty: str = 'easy'  # "easy", "medium", "hard"
    num_steps: int = 8  # Steps per rollout. 8 for easy, and 16 for medium
    repeatable: bool = False
    step_penalty: float = 0.1
    history_len: int = 3

    # PPO hyperparameters
    batch_size: int = 4
    accumulation_steps: int = 16
    learning_rate: float = 1e-5
    learning_rate_value_head: float = 1e-4
    epsilon_clip: float = 0.2
    ppo_epochs: int = 4
    gamma: float = 0.9
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    # Training settings
    num_iterations: int = 200
    save_interval: int = 50
    log_interval: int = 10
    eval_interval: int = 5

    # Loss settings
    value_loss_coef: float = 2.0
    entropy_coef: float = 0.05
    kl_coef: float = 0.1  # Start with 0.1, tune based on results

    # LoRA settings
    lora_enabled: bool = False  # Enable LoRA
    lora_rank: int = 16  # Low-rank dimension for LoRA
    lora_dropout: float = 0.1  # Dropout for LoRA layers

    # Exploration
    epsilon: float = 0.0
    epsilon_decay: float = 0.95
    min_epsilon: float = 0.0
    temperature: float = 5.0
    min_temperature: float = 1.0
    temperature_decay: float = 0.9


    # Evaluation
    num_eval_episodes: int = 20
    num_sample_games: int = 5

    # KL Penalty settings
    use_kl_penalty: bool = True  # ONLY works with action_token
    reference_fp16: bool = True  # Use FP16 for reference model to save memory
