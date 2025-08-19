# evaluator.py
import torch
import numpy as np
import wandb
from typing import Tuple

from .experience_roller import ExperienceRoller

class Evaluator:
    """
    Handles policy evaluation by running a fixed set of N episodes.
    This is a wrapper around the unified ExperienceRoller.
    """
    def __init__(self, policy, config, device, logger):
        self.config = config
        self.device = device
        self.logger = logger
        self.roller = ExperienceRoller(policy, config, device)
        self.num_envs = config.num_evals

    def run_evaluation(self, iteration: int) -> Tuple[float, float]:
        """
        Run N parallel evaluation rollouts.

        Args:
            iteration (int): The current training iteration number.

        Returns:
            A tuple containing the average episode length and average episode reward.
        """
        self.logger.info(f"\n\nRunning fixed-set parallel evaluation with {self.num_envs} environments...")
        
        # Use the unified roller in "evaluation" mode (low temperature for greedy actions)
        _, finished_episodes = self.roller.run(
            num_episodes=self.num_envs,
            temperature=0.01, # Set low for greedy/deterministic behavior
            is_eval_mode=True
        )

        # --- Reporting ---
        avg_episode_length = np.mean([ep.length for ep in finished_episodes])
        avg_episode_reward = np.mean([ep.reward for ep in finished_episodes])
        completed_count = sum(1 for ep in finished_episodes if ep.completed)
        
        self.logger.info(
            f"Evaluation complete: {len(finished_episodes)} total episodes run, "
            f"{completed_count} won, "
            f"Avg Length: {avg_episode_length:.2f}, "
            f"Avg Reward: {avg_episode_reward:.4f}"
        )

        # Log sample games to file
        sample_games_logs = sorted([ep.game_log for ep in finished_episodes if ep.game_log is not None])
        if sample_games_logs:
            all_games_text = "\n\n".join(sample_games_logs)
            with open(f"evaluations/{wandb.run.name}.txt", "a") as f:
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Evaluation Metrics - Total Episodes: {len(finished_episodes)}, ")
                f.write(f"Completed (Won): {completed_count}, ")
                f.write(f"Avg Length: {avg_episode_length:.2f}, Avg Reward: {avg_episode_reward:.4f}\n\n")
                f.write(all_games_text)
                f.write("\n\n")

        torch.cuda.empty_cache()

        return avg_episode_length, avg_episode_reward