import torch
import numpy as np
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from torch.amp import autocast
import psutil
import gc


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
        all_episode_lengths = []
        all_episode_rewards = []

        self.logger.info(f"Collecting rollouts for {self.config.num_steps} steps...")
        
        # Monitor memory usage
        initial_memory = self._get_memory_usage()
        self.logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        for step in tqdm(range(self.config.num_steps), desc="Collecting"):
            # Memory monitoring
            if step % 10 == 0:  # Check every 10 steps
                current_memory = self._get_memory_usage()
                if current_memory > initial_memory * 2:  # Memory doubled
                    self.logger.warning(f"Memory usage high: {current_memory:.2f} MB at step {step}")
                    self._force_cleanup()

            # Get valid actions for all environments
            batch_states = []
            batch_actions = []

            for i, (env, state) in enumerate(zip(self.envs, states)):
                actions = env.get_valid_actions()
                if not actions:
                    actions = ["look", "inventory", "help"]  # Fallback actions
                
                batch_states.append(state)
                batch_actions.append(actions)

            # Evaluate actions
            with torch.no_grad(), autocast(self.device.type):
                action_logprobs, values = self.policy.evaluate_for_rollout(
                    batch_states, batch_actions
                )

            # Step each environment
            new_states = []
            step_experiences = []
            action_prompts_batch = []
            
            for i, (env, state, actions, logprobs, value) in enumerate(
                zip(self.envs, states, batch_actions, action_logprobs, values)
            ):
                # Select action (epsilon-greedy)
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, len(actions) - 1)
                else:
                    action_idx = np.argmax(logprobs)

                chosen_action = actions[action_idx]
                
                # Take step
                next_state, reward, done, info = env.step(chosen_action)

                # Update episode tracking
                episode_lengths[i] += 1
                episode_rewards[i] += reward

                # Check if this is the last step (truncation)
                is_last_step = step == self.config.num_steps - 1
                truncated = done or is_last_step

                # Store experience
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
                action_prompt = f"In game state: {state}, best action is {chosen_action}"
                action_prompts_batch.append(action_prompt)

                # Update state and episode tracking
                if not done:
                    updated_state = (
                        state + "\n\n" + 
                        f"action taken: {chosen_action}\n\n" + 
                        next_state
                    )
                    new_states.append(updated_state)
                else:
                    # Store completed episode metrics
                    all_episode_lengths.append(episode_lengths[i])
                    all_episode_rewards.append(episode_rewards[i])
                    # Reset episode tracking
                    episode_lengths[i] = 0
                    episode_rewards[i] = 0.0
                    episode_ids[i] += 1
                    new_states.append(env.reset())
            
            # Process action prompts in batch
            if step_experiences and action_prompts_batch:
                with torch.no_grad(), autocast(self.device.type):
                    action_inputs = self.policy.tokenize_prompts(action_prompts_batch)
                    action_inputs = {k: v.to(self.device) for k, v in action_inputs.items()}
                    batch_logits, _ = self.policy(**action_inputs)

                    # Store experiences without old_logits to avoid memory leak
                    for i, (experience, logits) in enumerate(
                        zip(step_experiences, batch_logits)
                    ):
                        # OLD LOGIC (causes memory leak):
                        # experience["old_logits"] = logits.cpu().detach().half()
                        # del logits
                        rollout_buffer.append(experience)

                # Clean up GPU memory
                del action_inputs, batch_logits
                torch.cuda.empty_cache()
                gc.collect()

            states = new_states

        # Store metrics for truncated episodes
        for i, (length, reward) in enumerate(zip(episode_lengths, episode_rewards)):
            if length > 0:
                all_episode_lengths.append(length)
                all_episode_rewards.append(reward)

        final_memory = self._get_memory_usage()
        self.logger.info(f"Final memory usage: {final_memory:.2f} MB")

        return rollout_buffer, all_episode_lengths, all_episode_rewards

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _force_cleanup(self):
        """Force garbage collection and GPU memory cleanup"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_epsilon(self, new_epsilon: float):
        """Update epsilon for epsilon-greedy exploration"""
        self.epsilon = new_epsilon