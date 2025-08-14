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
        self.epsilon = config.epsilon

    def collect_rollouts(self) -> Tuple[List[Dict], List[int], List[float]]:
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
                action_logprobs, values = self.policy.evaluate_for_rollout(
                    batch_states, batch_actions
                )

            # Step each environment
            new_states = []
            
            for i, (env, state, actions, logprobs, value) in enumerate(
                zip(self.envs, states, batch_actions, action_logprobs, values)
            ):
                # Select action (epsilon-greedy)
                # if random.random() < self.epsilon:
                #     action_idx = random.randint(0, len(actions) - 1)
                # else:
                #     action_idx = np.argmax(logprobs)

                action_idx, old_logprob_indexed = self.temperature_sampling_with_floor(np.array(logprobs))
                

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

    def update_epsilon(self, new_epsilon: float):
        """Update epsilon for epsilon-greedy exploration"""
        self.epsilon = new_epsilon
    
    def temperature_sampling_with_floor(self, logprobs):
        # Convert from log-probs to probs
        probs = np.exp(logprobs)
        
        # Apply temperature
        probs = probs ** (1 / self.config.sampling_temperature)
        probs /= probs.sum()

        # Apply hard floor to every action's probability
        probs = np.maximum(probs, self.config.softmax_floor)
        probs /= probs.sum()

        # Sample action
        action_idx = np.random.choice(len(probs), p=probs)

        # Get the log-prob from the modified distribution
        old_logprob = np.log(probs[action_idx] + 1e-8)

        return action_idx, old_logprob