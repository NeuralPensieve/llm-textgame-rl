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

        episode_ids = [0] * len(self.envs)
        episode_lengths = [0] * len(self.envs)
        episode_rewards = [0.0] * len(self.envs)
        episode_step_counts = [0] * len(self.envs)
        all_episode_lengths = []
        all_episode_rewards = []

        self.logger.info(f"Collecting rollouts for {self.config.num_steps} steps...")
        
        for step in tqdm(range(self.config.num_steps), desc="Collecting"):
            # Get valid actions for all environments
            batch_states = []
            batch_actions = []

            for env, state in zip(self.envs, states):
                actions = env.get_valid_actions()
                batch_states.append(state)
                batch_actions.append(actions)

            # Evaluate actions
            with torch.no_grad(), autocast(self.device.type):
                action_scores, values = self.policy.evaluate_for_rollout(
                    batch_states, batch_actions
                )

            # Step each environment
            new_states = []
            
            for i, (env, state, actions, scores, value) in enumerate(
                zip(self.envs, states, batch_actions, action_scores, values)
            ):
                # Get action and original logprob from the new sampling method
                action_idx, old_logprob_indexed = self.temperature_sampling(scores, temperature)
                
                chosen_action = actions[action_idx]
                
                # Take step in environment
                next_state, reward, done, info = env.step(chosen_action)

                # Update episode tracking
                episode_lengths[i] += 1
                episode_rewards[i] += reward
                episode_step_counts[i] += 1

                # Check if this is the last step (truncation)
                is_last_step = step == self.config.num_steps - 1
                truncated = done or is_last_step

                # Store experience
                rollout_buffer.append({
                    "env_idx": i,
                    "episode_id": episode_ids[i],
                    "state": state,
                    "action": chosen_action,
                    "action_idx": action_idx,
                    "available_actions": actions,
                    "old_logprob": old_logprob_indexed,
                    "value": value,
                    "reward": reward,
                    "done": done,
                    "truncated": truncated,
                })

                # Check if episode reached natural conclusion
                # TODO: Fix end of step issue
                episode_concluded = done or (episode_step_counts[i] >= self.config.num_steps)
                
                # Update state for next step
                if episode_concluded:
                    # Episode reached natural conclusion - store metrics and reset
                    all_episode_lengths.append(episode_lengths[i])
                    all_episode_rewards.append(episode_rewards[i])
                    episode_lengths[i] = 0
                    episode_rewards[i] = 0.0
                    episode_step_counts[i] = 0
                    episode_ids[i] += 1
                    new_states.append(env.reset())
                else:
                    # Continue episode
                    updated_state = (
                        state + "\n\n" + 
                        f"action taken: {chosen_action}\n\n" + 
                        next_state
                    )
                    new_states.append(updated_state)
            
            # Update states for next iteration
            states = new_states

        return rollout_buffer, all_episode_lengths, all_episode_rewards
    
    def temperature_sampling(self, raw_scores, temperature):
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
        
        # Apply temperature for sampling distribution
        scaled_scores = scores / temperature
        
        # Create sampling probabilities with numerical stability
        # Subtracting the max score prevents overflow when exponentiating
        scaled_scores_stable = scaled_scores - np.max(scaled_scores)
        sampling_probs = np.exp(scaled_scores_stable)
        sampling_probs /= sampling_probs.sum()
        
        # Ensure probabilities are normalized due to potential floating point inaccuracies
        sampling_probs = sampling_probs / np.sum(sampling_probs)

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