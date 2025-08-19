import numpy as np
from typing import List, Tuple

from .experience_roller import ExperienceRoller

class RolloutCollector:
    """
    Handles experience collection by running full episodes.
    This is a wrapper around the unified ExperienceRoller.
    """
    def __init__(self, policy, config, device, logger):
        self.config = config
        self.logger = logger
        self.roller = ExperienceRoller(policy, config, device)
        self.num_envs = config.num_envs # Number of parallel envs for training

    def collect_rollouts(self, temperature: float) -> Tuple[List[dict], List[int], List[float]]:
        """
        Collect experience from environments by running a batch of episodes to completion.
        
        Args:
            temperature (float): The temperature for sampling actions.

        Returns:
            A tuple containing the rollout buffer, a list of all episode lengths, 
            and a list of all episode rewards.
        """
        self.logger.info(f"Collecting rollouts from {self.num_envs} parallel environments (episode-centric)...")

        # Use the unified roller in "training" mode
        rollout_buffer, finished_episodes = self.roller.run(
            num_episodes=self.num_envs,
            temperature=temperature,
            is_eval_mode=False
        )

        # Extract stats from the results
        all_episode_lengths = [ep.length for ep in finished_episodes]
        all_episode_rewards = [ep.reward for ep in finished_episodes]
        
        total_steps = len(rollout_buffer)
        avg_len = np.mean(all_episode_lengths) if all_episode_lengths else 0
        avg_rew = np.mean(all_episode_rewards) if all_episode_rewards else 0

        self.logger.info(f"Collection complete. Total steps: {total_steps}. "
                         f"Completed episodes: {len(finished_episodes)}. "
                         f"Avg Length: {avg_len:.2f}, Avg Reward: {avg_rew:.4f}")

        return rollout_buffer, all_episode_lengths, all_episode_rewards