import os
import torch
import logging
import datetime
import wandb
from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """Abstract base class for all trainers"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iteration = 0

        # Setup logging and wandb
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging and wandb"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        run_name = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        wandb.init(project="textworld-llm-ppo", name=run_name, config=vars(self.config))

        # Create a folder, if doesn't exist, for evaluations with run_name
        os.makedirs("evaluations", exist_ok=True)

    @abstractmethod
    def train(self):
        """Main training loop - must be implemented by subclasses"""
        pass

    @abstractmethod
    def save_checkpoint(self, iteration: int):
        """Save training checkpoint - must be implemented by subclasses"""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint - must be implemented by subclasses"""
        pass
