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
        states = [env.reset() for env in self.envs]

        episode_ids = [0] * len(self.envs)  # Track current episode per env

        episode_lengths = [0] * len(self.envs)  # Track steps per episode
        episode_rewards = [0.0] * len(self.envs)  # Track total reward per episode
        all_episode_lengths = []  # Store completed episode lengths
        all_episode_rewards = []  # Store completed episode rewards

        self.logger.info(f"Collecting rollouts for {self.config.num_steps} steps...")

        for step in tqdm(range(self.config.num_steps), desc="Collecting"):
            # Get valid actions for all environments
            batch_states = []
            batch_actions = []
            env_indices = []

            for i, (env, state) in enumerate(zip(self.envs, states)):
                actions = env.get_valid_actions()
                batch_states.append(state)
                batch_actions.append(actions)
                env_indices.append(i)

            # Evaluate actions with no cache
            with torch.no_grad(), autocast(self.device.type):
                action_logprobs, values = self.policy.evaluate_for_rollout(
                    batch_states, batch_actions
                )

            # Step each environment
            new_states = []
            step_experiences = []  # Store experiences for this step
            action_prompts_batch = []  # Collect all action prompts for batch processing
            for i, (env, state, actions, logprobs, value) in enumerate(
                zip(self.envs, states, batch_actions, action_logprobs, values)
            ):

                # Select action (epsilon-greedy)
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, len(actions) - 1)
                else:
                    action_idx = np.argmax(logprobs)

                chosen_action = actions[action_idx]
                next_state, reward, done, info = env.step(chosen_action)

                # Update episode tracking
                episode_lengths[i] += 1  # Increment step count
                episode_rewards[i] += reward  # Accumulate reward

                # Check if this is the last step (truncation)
                is_last_step = step == self.config.num_steps - 1
                truncated = done or is_last_step

                # Store experience (without old_logits for now)
                experience = {
                    "env_idx": i,
                    "episode_id": episode_ids[i],
                    "state": state,
                    "action": chosen_action,
                    "action_idx": action_idx,
                    "available_actions": actions,
                    "old_logprob": logprobs[action_idx],
                    "value": value,
                    "reward": reward,
                    "done": done,
                    "truncated": truncated,
                }
                step_experiences.append(experience)

                # Collect action prompt for batch processing
                action_prompt = (
                    f"In game state: {state}, best action is {chosen_action}"
                )
                action_prompts_batch.append(action_prompt)

                # Update state and episode tracking
                if not done:
                    # Continue episode with action context
                    updated_state = (
                        state
                        + "\n\n"
                        + f"action taken: {chosen_action}\n\n"
                        + next_state
                    )
                    new_states.append(updated_state)
                else:
                    # Store completed episode metrics
                    all_episode_lengths.append(episode_lengths[i])
                    all_episode_rewards.append(episode_rewards[i])
                    # Reset episode tracking
                    episode_lengths[i] = 0
                    episode_rewards[i] = 0.0
                    # Episode ended, increment counter and reset environment
                    episode_ids[i] += 1
                    new_states.append(env.reset())

            # Process all action prompts in batch to ensure consistent padding
            with torch.no_grad(), autocast(self.device.type):
                # Tokenize all action prompts together (same padding as training)
                action_inputs = self.policy.tokenize_prompts(action_prompts_batch)
                action_inputs = {k: v.to(self.device) for k, v in action_inputs.items()}

                # Get logits for all actions in batch
                batch_logits, _ = self.policy(**action_inputs)

                # Store individual logits back to experiences
                for i, (experience, logits) in enumerate(
                    zip(step_experiences, batch_logits)
                ):
                    experience["old_logits"] = logits.cpu()  # Store individual logits
                    rollout_buffer.append(experience)

            # Clean up GPU memory
            del action_inputs, batch_logits
            torch.cuda.empty_cache()

            states = new_states

        # Store metrics for truncated episodes
        for i, (length, reward) in enumerate(zip(episode_lengths, episode_rewards)):
            if length > 0:  # Only include non-zero length episodes
                all_episode_lengths.append(length)
                all_episode_rewards.append(reward)

        # Return additional metrics
        return rollout_buffer, all_episode_lengths, all_episode_rewards

    def update_epsilon(self, new_epsilon: float):
        """Update epsilon for epsilon-greedy exploration"""
        self.epsilon = new_epsilon
