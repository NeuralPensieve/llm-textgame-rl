# TextWorld LLM-PPO Training

This project implements a token-level Proximal Policy Optimization (PPO) training pipeline for Language Models (LLMs) in the TextWorld environment. It leverages PyTorch, Hugging Face Transformers, and Weights & Biases (WandB) for experiment tracking.

A detailed blog post about this work can be found here: [https://neuralpensieve.github.io/2025/08/26/rl-llm-textworld.html](https://neuralpensieve.github.io/2025/08/26/rl-llm-textworld.html)

## Overview

The project is structured into several key modules:

-   **`config/`**: Defines configuration classes for PPO training, including static and dynamic parameters.
    -   [`config/ppo_config.py`](config/ppo_config.py): Defines the main PPO configuration class ([`PPOConfig`](config/ppo_config.py)).
-   **`env/`**: Contains the TextWorld environment wrapper.
    -   `env/textworld_env.py`: Implements the TextWorld environment interface ([`TextWorldEnvironment`](env/textworld_env.py)).
-   **`helper/`**: Provides helper functions for tokenization, Trie data structure, and prompt formatting.
    -   [`helper/trie.py`](helper/trie.py): Implements the Trie data structure for action masking and related functions ([`Trie`](helper/trie.py), [`generate_mask`](helper/trie.py), [`tokenize_actions_for_trie`](helper/trie.py)).
    -   [`helper/tokenizer.py`](helper/tokenizer.py): Manages tokenization using Hugging Face Transformers ([`TokenizerHelper`](helper/tokenizer.py)).
    -   [`helper/prompt_manager.py`](helper/prompt_manager.py): Formats the input prompts for the language model ([`format_prompt`](helper/prompt_manager.py)).
-   **`models/`**: Defines the LLM policy model.
    -   `models/llm_policy.py`: Implements the LLM policy model ([`LLMPolicy`](models/llm_policy.py)).
-   **`trainer/`**: Contains the core training components.
    -   [`trainer/base_trainer.py`](trainer/base_trainer.py): Defines the abstract base class for trainers ([`BaseTrainer`](trainer/base_trainer.py)).
    -   [`trainer/rollout_collector.py`](trainer/rollout_collector.py): Collects experiences from the environment using the current policy ([`RolloutCollector`](trainer/rollout_collector.py)).
    -   [`trainer/experience_roller.py`](trainer/experience_roller.py): Manages parallel environments and generates experiences ([`ExperienceRoller`](trainer/experience_roller.py), [`ParallelEnvManager`](trainer/experience_roller.py), [`ActionGenerator`](trainer/experience_roller.py), [`EpisodeStats`](trainer/experience_roller.py), [`GenerationResult`](trainer/experience_roller.py)).
    -   [`trainer/ppo_updater.py`](trainer/ppo_updater.py): Implements the PPO update logic ([`PPOUpdater`](trainer/ppo_updater.py)).
    -   [`trainer/ppo_trainer.py`](trainer/ppo_trainer.py): Orchestrates the training process, including evaluation, rollout collection, and policy updates ([`PPOTextWorldTrainer`](trainer/ppo_trainer.py)).
-   **`tests/`**: Includes unit and integration tests for various components.
    -   [`tests/mocks/`](tests/mocks/): Contains mock classes for testing.
    -   [`tests/integration/`](tests/integration/): Contains integration tests.
-   **`train.py`**: The main script for launching the training process.
-   **`test.py`**: A script for playing a TextWorld game interactively.

## Key Components

### Experience Roller

The [`ExperienceRoller`](trainer/experience_roller.py) is responsible for running the TextWorld environments and collecting experiences. It manages parallel environments using the [`ParallelEnvManager`](trainer/experience_roller.py) and generates actions using the [`ActionGenerator`](trainer/experience_roller.py). The `run` method orchestrates the rollout generation for training or evaluation.

### Action Generator

The [`ActionGenerator`](trainer/experience_roller.py) handles the token-by-token generation of actions using the policy model. It uses a Trie data structure to constrain the action space and ensures that only valid actions are generated.

### PPO Trainer

The [`PPOTextWorldTrainer`](trainer/ppo_trainer.py) is the main trainer class that orchestrates the entire training process. It initializes the policy, optimizer, scheduler, and other components. The `train` method implements the main training loop, which includes evaluation, rollout collection, and policy updates.

## Training Process

The training process consists of the following steps:

1.  **Initialization**: The trainer initializes the policy model, optimizer, scheduler, and other components.
2.  **Evaluation**: The policy is evaluated periodically to track its performance.
3.  **Rollout Collection**: The [`RolloutCollector`](trainer/rollout_collector.py) collects experiences from the environment using the current policy.
4.  **PPO Update**: The [`PPOUpdater`](trainer/ppo_updater.py) updates the policy using the collected experiences.
5.  **Temperature Decay**: The temperature parameter is decayed to reduce exploration over time.
6.  **Checkpointing**: The trainer saves checkpoints periodically to preserve the training progress.

## Usage

### Training

To start the training process, run the `train.py` script:

```bash
python train.py
```

You can also provide a checkpoint file to resume training from a previous state:

```bash
python train.py --checkpoint checkpoints/run_name_iter_100.pt
```

### Interactive Play
To play a TextWorld game interactively, run the `interactive_test.py` script:

```bash
python interactive_test.py
```

You can specify the difficulty level and random seed for the game:

```bash
python test.py --difficulty medium --seed 42
```

## Installation
Before running the project, ensure you have the following installed:

Python: Version 3.7 or higher.
PyTorch: Follow the installation instructions on the PyTorch website to install the correct version for your system. Make sure you have a compatible CUDA version installed if you plan to use a GPU.
CUDA: (Optional) If you have a NVIDIA GPU, install CUDA to accelerate training.
Dependencies
The project depends on the following Python libraries:

- torch
- numpy
- transformers
- wandb
- bitsandbytes
- textworld

You can install these dependencies using pip:

```bash
pip install torch numpy transformers wandb bitsandbytes textworld
```

## Citation

If you use this code in your research or project, please cite it as follows:

### BibTeX
```bibtex
@software{textworld_llm_ppo,
  title={TextWorld LLM-PPO Training},
  author={[Ali Roshan-Ghias]},
  year={[2025]},
  url={[https://github.com/NeuralPensieve/llm-textgame-rl]},
  note={A token-level PPO training pipeline for Language Models in TextWorld environments}
}
```

## License
This project is licensed under the MIT License.
