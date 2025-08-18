import torch
import numpy as np
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from torch.amp import autocast


class RolloutCollector:
    """Handles experience collection from environments"""

    def __init__(self, policy, envs, config, device, logger):
        self.policy = policy
        self.envs = envs
        self.config = config
        self.device = device
        self.logger = logger

    def collect_rollouts(self, temperature: float, epsilon: float) -> Tuple[List[Dict], List[int], List[float]]:
        """Collect experience from environments"""        
        rollout_buffer = []
        
        # Initialize states
        states = [env.reset() for env in self.envs]

        episode_rewards = [0.0] * len(self.envs)
        episode_lengths = [0] * len(self.envs)
        all_episode_rewards = []
        all_episode_lengths = []

        self.logger.info(f"Collecting rollouts for {self.config.num_steps} steps...")
        
        for step in tqdm(range(self.config.num_steps), desc="Collecting Rollouts"):
            # Get valid actions for all environments
            batch_states = []
            batch_actions = []

            for env, state in zip(self.envs, states):
                actions = env.get_valid_actions()
                batch_states.append(state)
                batch_actions.append(actions)

            with torch.no_grad(), autocast(self.device.type):
                action_scores, values = self.policy.evaluate_for_rollout(batch_states, batch_actions)

            new_states = [None] * len(self.envs)
            
            for i, env in enumerate(self.envs):
                actions = batch_actions[i]
                
                # Use the provided temperature sampling method
                action_idx, old_logprob_indexed = self.temperature_sampling(action_scores[i], temperature)
                chosen_action = actions[action_idx] if actions else "wait"
                
                next_state, reward, done, info = env.step(chosen_action)

                episode_rewards[i] += reward
                episode_lengths[i] += 1

                rollout_buffer.append({
                    "env_idx": i,
                    "state": states[i],
                    "action": chosen_action,
                    "action_idx": action_idx,
                    "available_actions": actions,
                    "old_logprob": old_logprob_indexed,
                    "value": values[i],
                    "reward": reward,
                    "done": done,
                })

                # --- CORRECTED RESET AND STATE UPDATE LOGIC ---
                if done:
                    all_episode_rewards.append(episode_rewards[i])
                    all_episode_lengths.append(episode_lengths[i])
                    
                    # Reset the environment and use the new state for the next step
                    new_states[i] = env.reset()
                    
                    # Reset trackers for the new episode
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                else:
                    # Continue the current episode
                    new_states[i] = next_state
            
            states = new_states

        return rollout_buffer, all_episode_lengths, all_episode_rewards
    
    def temperature_sampling(self, raw_scores, temperature: float) -> Tuple[int, float]:
        """
        Apply temperature sampling to raw action scores with numerical stability.
        
        Args:
            raw_scores: Raw action scores (logits) from the policy.
            temperature: Temperature parameter for sampling.
        
        Returns:
            action_idx: The selected action index.
            old_logprob: The log probability of the selected action from the original policy.
        """
        scores = np.array(raw_scores)

        # Prevent errors on empty scores list
        if not scores.any():
            return 0, -np.inf
        
        # Apply temperature for sampling distribution
        scaled_scores = scores / temperature
        
        # Create sampling probabilities with numerical stability
        # Subtracting the max score prevents overflow when exponentiating
        scaled_scores_stable = scaled_scores - np.max(scaled_scores)
        sampling_probs = np.exp(scaled_scores_stable) / np.sum(np.exp(scaled_scores_stable))

        # Sample an action from the modified distribution
        action_idx = np.random.choice(len(sampling_probs), p=sampling_probs)
        
        # Now, calculate the log probability from the ORIGINAL, non-scaled scores
        # This is the log-softmax function, also with a stability trick
        original_scores_stable = scores - np.max(scores)
        exp_scores = np.exp(original_scores_stable)
        log_sum_exp = np.log(np.sum(exp_scores))
        original_logprobs = original_scores_stable - log_sum_exp
        
        old_logprob = original_logprobs[action_idx]
        
        return action_idx, old_logprob