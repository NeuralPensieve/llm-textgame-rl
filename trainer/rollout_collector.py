import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

# Assuming these are in your project structure
from .experience_roller import ExperienceRoller, EpisodeStats
from models import LLMPolicy
from config import PPOConfig
from helper import TokenizerHelper

class RolloutCollector:
    """
    Handles experience collection by running full episodes. It wraps the
    ExperienceRoller and can be used for both training and evaluation.
    """
    def __init__(self, policy: LLMPolicy, config: PPOConfig, device: torch.device, logger, tokenizer_helper: TokenizerHelper):
        self.config = config
        self.logger = logger
        # Pass the tokenizer_helper down to the roller
        self.roller = ExperienceRoller(policy, config, device, tokenizer_helper)
        self.num_envs = config.num_envs # Number of parallel envs for training

    def collect_rollouts(self, temperature: float, is_eval_mode: bool = False) -> Tuple[List[Dict], List[EpisodeStats]]:
        """
        Collects experience by running a batch of episodes to completion.
        
        Args:
            temperature (float): The temperature for sampling actions.
            is_eval_mode (bool): Flag to determine if this is for evaluation.

        Returns:
            A tuple containing the token-level rollout buffer (empty if eval)
            and a list of EpisodeStats for each completed episode.
        """
        mode = "Evaluation" if is_eval_mode else "Training Rollout"
        num_episodes = self.config.num_eval_episodes if is_eval_mode else self.num_envs
        self.logger.info(f"Starting {mode} with {num_episodes} parallel environments...")

        rollout_buffer, finished_episodes = self.roller.run(
            num_episodes=num_episodes,
            temperature=temperature,
            is_eval_mode=is_eval_mode
        )

        # Log the game logs if in evaluation mode
        # if is_eval_mode:
        #     for episode_stat in finished_episodes:
        #         if episode_stat.game_log:
        #             self.logger.info(episode_stat.game_log)

        # Post-process to get final stats for logging
        if not finished_episodes:
             self.logger.warning(f"{mode} resulted in no completed episodes.")
             return rollout_buffer, []

        all_episode_lengths = [ep.length for ep in finished_episodes]
        all_episode_rewards = [ep.reward for ep in finished_episodes]
        
        total_steps = len(rollout_buffer)
        avg_len = np.mean(all_episode_lengths)
        avg_rew = np.mean(all_episode_rewards)

        self.logger.info(f"{mode} complete. Total token steps: {total_steps}. "
                         f"Completed episodes: {len(finished_episodes)}. "
                         f"Avg Length (tokens): {avg_len:.2f}, Avg Reward: {avg_rew:.4f}")

        return rollout_buffer, finished_episodes