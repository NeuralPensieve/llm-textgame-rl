"""
Trainer package for PPO TextWorld training.

This package contains modular components for PPO training:
- BaseTrainer: Abstract base class with common functionality
- RolloutCollector: Handles experience collection from environments
- Evaluator: Handles policy evaluation and sample game generation
- PPOUpdater: Handles PPO policy updates and advantage computation
- PPOTextWorldTrainer: Main trainer class that orchestrates all components
"""

from .base_trainer import BaseTrainer
from .rollout_collector import RolloutCollector
from .evaluator import Evaluator
from .ppo_updater import PPOUpdater
from .ppo_trainer import PPOTextWorldTrainer

__all__ = [
    "BaseTrainer",
    "RolloutCollector",
    "Evaluator",
    "PPOUpdater",
    "PPOTextWorldTrainer",
]
