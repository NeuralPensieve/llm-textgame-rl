import torch
import numpy as np
import datetime
import os
import wandb
from torch.amp import GradScaler

from env import TextWorldEnvironment
from config import PPOConfig
from models import LLMPolicy
from trainer import BaseTrainer, RolloutCollector, Evaluator, PPOUpdater


class PPOTextWorldTrainer(BaseTrainer):
    """Simplified PPO trainer for TextWorld"""

    def __init__(self, config: PPOConfig):
        super().__init__(config)
        
        # Clear games folder
        if not os.path.exists("./games"):
            os.makedirs("./games")
        else:
            if not self.config.reuse_seed:
                for f in os.listdir("./games"):
                    os.remove(os.path.join("./games", f))

        # Create environments and policy
        self.envs = [TextWorldEnvironment(config=config) for _ in range(config.num_envs)]
        self.policy = LLMPolicy(config).to(self.device)

        # Disable cache for transformer models during training
        if hasattr(self.policy, "config"):
            self.policy.config.use_cache = False

        # Improved optimizer with separate learning rates
        param_groups = self.policy.get_separate_parameter_groups()
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_iterations, eta_min=1e-7
        )

        # Initialize GradScaler for FP16 training
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # Training state
        self.epsilon = config.epsilon
        self.temperature = config.temperature

        os.makedirs("checkpoints", exist_ok=True)

        # Initialize components
        self.rollout_collector = RolloutCollector(
            self.policy, self.envs, config, self.device, self.logger
        )
        # torch.cuda.empty_cache()

        self.evaluator = Evaluator(self.policy, config, self.device, self.logger)
        # torch.cuda.empty_cache()

        self.ppo_updater = PPOUpdater(
            config,
            self.policy,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.device,
            self.logger,
        )
        # torch.cuda.empty_cache()

    def train(self):
        """Main training loop"""
        self.logger.info("Starting PPO training...")

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration

            # Start timing the iteration
            start_time = datetime.datetime.now()

            # Run evaluation
            if iteration % self.config.eval_interval == 0:
                eval_length, eval_reward = self.evaluator.run_evaluation(iteration, self.temperature)
                wandb.log(
                    {
                        "iteration": iteration,
                        "eval_avg_episode_length": eval_length,
                        "eval_avg_episode_reward": eval_reward,
                    }
                )

            # Collect rollouts
            rollout_buffer, episode_lengths, episode_rewards = (
                self.rollout_collector.collect_rollouts(temperature=self.temperature, epsilon=self.epsilon)
            )

            # Calculate metrics
            rewards = [exp["reward"] for exp in rollout_buffer]
            avg_reward = np.mean(rewards)
            avg_episode_length = np.mean(episode_lengths)
            avg_episode_reward = np.mean(episode_rewards)
            total_episode_reward = np.sum(episode_rewards)

            # Update policy
            self.ppo_updater.ppo_update(rollout_buffer, iteration, self.temperature)

            # Decay epsilon
            self.epsilon = max(
                self.config.min_epsilon, self.epsilon * self.config.epsilon_decay
            )

            # Decay temperature
            self.temperature = max(
                self.config.min_temperature, self.temperature * self.config.temperature_decay
            )

            # Calculate iteration duration and normalize by rollout size
            end_time = datetime.datetime.now()
            iteration_duration = (
                end_time - start_time
            ).total_seconds()  # Duration in seconds
            rollout_size = (
                len(rollout_buffer) if rollout_buffer else 1
            )  # Avoid division by zero
            normalized_duration = iteration_duration / rollout_size

            # Log iteration metrics
            wandb.log(
                {
                    "iteration": iteration,
                    "avg_reward": avg_reward,
                    "avg_episode_length": avg_episode_length,
                    "avg_episode_reward": avg_episode_reward,
                    "total_episode_reward": total_episode_reward,
                    "total_experiences": len(rollout_buffer),
                    "epsilon": self.epsilon,
                    "temperature": self.temperature,
                    "iteration_duration": iteration_duration,
                    "normalized_iteration_duration": normalized_duration,
                }
            )

            # Save checkpoint
            if (iteration + 1) % self.config.save_interval == 0 and iteration > 10:
                self.save_checkpoint(iteration)

        wandb.finish()
        self.logger.info("Training completed!")

    def save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "epsilon": self.epsilon,
        }

        run_name = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        checkpoint_path = f"checkpoints/{run_name}_iter_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        # Fix for PyTorch 2.6+ - set weights_only=False for trusted checkpoints
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state if available
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.iteration = checkpoint["iteration"]
        self.epsilon = checkpoint["epsilon"]

        # Update components with loaded state
        self.rollout_collector.update_epsilon(self.epsilon)
        self.rollout_collector.update_temperature(self.temperature)

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
