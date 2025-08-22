import torch
import numpy as np
import datetime
import os
import wandb
import bitsandbytes as bnb
from torch.amp import GradScaler

from config import PPOConfig
from models import LLMPolicy
from trainer import BaseTrainer, RolloutCollector, PPOUpdater
from helper import TokenizerHelper


class PPOTextWorldTrainer(BaseTrainer):
    """The main trainer for running token-level PPO on TextWorld."""

    def __init__(self, config: PPOConfig):
        super().__init__(config)

        self.tokenizer_helper = TokenizerHelper(config)

        # Create policy
        self.policy = LLMPolicy(config, self.tokenizer_helper, self.device)

        # Disable cache for transformer models during training
        if hasattr(self.policy.model.config, "use_cache"):
            self.policy.model.config.use_cache = False

        # Optimizer with separate learning rates for model and value head
        param_groups = self.policy.get_separate_parameter_groups()
        # self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        self.optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.01)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_iterations, eta_min=1e-7
        )

        # Disable scaler if using full FP16
        if config.use_fp16:
            self.scaler = None  # Disable scaler for FP16
        else:
            # GradScaler for mixed-precision training
            self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # Training state
        self.temperature = config.temperature

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("evaluations", exist_ok=True)

        # Initialize components, passing the tokenizer_helper
        self.rollout_collector = RolloutCollector(
            self.policy, config, self.device, self.logger, self.tokenizer_helper
        )
        # The PPOUpdater will need to be updated separately to handle the new buffer format
        self.ppo_updater = PPOUpdater(
            config,
            self.policy,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.device,
            self.logger,
            self.tokenizer_helper
        )

    def train(self):
        """Main training loop"""
        self.logger.info("Starting PPO training with generative policy...")

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            start_time = datetime.datetime.now()

            # --- Run evaluation using the RolloutCollector ---
            if iteration % self.config.eval_interval == 0:
                self.logger.info(f"\n--- Starting Evaluation for Iteration {iteration} ---")
                # is_eval_mode=True ensures greedy sampling (temp=0) and detailed logging
                _, eval_finished_episodes = self.rollout_collector.collect_rollouts(
                    temperature=0.0, is_eval_mode=True
                )
                
                if eval_finished_episodes:
                    eval_lengths = [ep.length for ep in eval_finished_episodes]
                    eval_rewards = [ep.reward for ep in eval_finished_episodes]
                    wandb.log({
                        "iteration": iteration,
                        "eval_avg_episode_length": np.mean(eval_lengths),
                        "eval_avg_episode_reward": np.mean(eval_rewards),
                    })
                    self._log_evaluation_games_to_file(eval_finished_episodes, iteration)
                self.logger.info(f"--- Evaluation Finished ---")

            # --- Collect training rollouts ---
            rollout_buffer, finished_episodes = self.rollout_collector.collect_rollouts(
                temperature=self.temperature, is_eval_mode=False
            )

            # --- Calculate and log training metrics ---
            if finished_episodes:
                episode_lengths = [ep.length for ep in finished_episodes]
                episode_rewards = [ep.reward for ep in finished_episodes]

                all_action_log_probs = [
                    prob for ep in finished_episodes for prob in ep.action_log_probs
                ]
                avg_action_log_prob = np.mean(all_action_log_probs) if all_action_log_probs else 0.0
                
                wandb.log({
                    "iteration": iteration,
                    "avg_episode_length": np.mean(episode_lengths),
                    "avg_episode_reward": np.mean(episode_rewards),
                    "avg_action_log_prob": avg_action_log_prob,
                    "total_experiences_collected": len(rollout_buffer),
                    "temperature (decaying)": self.temperature,
                })

            # --- Update policy using the collected token-level experiences ---
            if rollout_buffer:
                self.ppo_updater.ppo_update(rollout_buffer, iteration)
            else:
                self.logger.warning("Rollout buffer is empty. Skipping PPO update.")

            # Decay temperature for exploration
            self.temperature = max(
                self.config.min_temperature, self.temperature * self.config.temperature_decay
            )

            # --- Log iteration timing and save checkpoint ---
            end_time = datetime.datetime.now()
            iteration_duration = (end_time - start_time).total_seconds()
            rollout_size = len(rollout_buffer) if rollout_buffer else 1
            
            wandb.log({
                "iteration": iteration,
                "iteration_duration": iteration_duration,
                "normalized_iteration_duration": iteration_duration / rollout_size,
            })

            if (iteration + 1) % self.config.save_interval == 0:
                self.save_checkpoint(iteration)

        wandb.finish()
        self.logger.info("Training completed!")

    def _log_evaluation_games_to_file(self, finished_episodes, iteration):
        """Writes the game logs from evaluation episodes to a text file."""
        sample_games_logs = sorted([ep.game_log for ep in finished_episodes if ep.game_log is not None])

        if not sample_games_logs:
            self.logger.info("No sample game logs were generated during evaluation.")
            return

        # Get a unique run name for the log file
        run_name = wandb.run.name if wandb.run else f"run_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        log_filename = os.path.join("evaluations", f"{run_name}.txt")

        # Combine all logs into a single string
        all_games_text = "\n\n".join(sample_games_logs)

        # Calculate summary statistics for the header
        avg_length = np.mean([ep.length for ep in finished_episodes])
        avg_reward = np.mean([ep.reward for ep in finished_episodes])
        completed_count = sum(1 for ep in finished_episodes if ep.completed)

        try:
            # Open in append mode 'a' to add logs from all iterations to the same file
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"--- Iteration: {iteration} ---\n")
                f.write(f"Evaluation Metrics - Total Episodes: {len(finished_episodes)}, ")
                f.write(f"Completed (Won): {completed_count}, ")
                f.write(f"Avg Length: {avg_length:.2f}, Avg Reward: {avg_reward:.4f}\n\n")
                f.write(all_games_text)
                f.write("\n\n" + "="*80 + "\n\n") # Separator for the next iteration
            self.logger.info(f"Saved {len(sample_games_logs)} evaluation game logs to {log_filename}")
        except IOError as e:
            self.logger.error(f"Failed to write evaluation logs to file: {e}")

    def save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": self.config,
            "temperature": self.temperature,
        }
        run_name = wandb.run.name if wandb.run else f"run_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        checkpoint_path = f"checkpoints/{run_name}_iter_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.temperature = checkpoint.get("temperature", self.config.temperature)
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path} at iteration {self.iteration}")