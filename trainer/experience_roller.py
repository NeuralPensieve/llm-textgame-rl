import torch
import numpy as np
from typing import Tuple, List, Dict, NamedTuple
from tqdm import tqdm
from torch.amp import autocast

from env import TextWorldEnvironment

class EpisodeStats(NamedTuple):
    length: int
    reward: float
    completed: bool
    game_log: str

class ExperienceRoller:
    """
    Handles running parallel environments until episodes complete.
    This is the unified, core logic for both training rollouts and evaluation.
    """

    def __init__(self, policy, config, device):
        self.policy = policy
        self.config = config
        self.device = device

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for x with numerical stability."""
        if not x.any(): 
            return np.array([])
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def _process_micro_batches(self, batch_states: List[str], batch_actions: List[List[str]]) -> Tuple[List, List]:
        """
        Processes a batch by breaking it into smaller micro-batches to conserve memory.
        """
        # Get micro_batch_size from config, with a sensible default.
        micro_batch_size = getattr(self.config, 'micro_batch_size', 16)
        
        # Pre-allocate lists to store aggregated results.
        all_action_scores = [[] for _ in range(len(batch_states))]
        all_values = [0.0] * len(batch_states)

        # Process the full batch in smaller chunks.
        for i in range(0, len(batch_states), micro_batch_size):
            # Create slices for the current micro-batch.
            mb_states = batch_states[i:i + micro_batch_size]
            mb_actions = batch_actions[i:i + micro_batch_size]
            
            if not mb_states:
                continue

            # Get model outputs for the small micro-batch.
            with torch.no_grad(), autocast(self.device.type):
                mb_scores, mb_values = self.policy.evaluate_for_rollout(mb_states, mb_actions)

            # Place the results from the micro-batch into the main result lists.
            for j in range(len(mb_states)):
                original_batch_idx = i + j
                all_action_scores[original_batch_idx] = mb_scores[j]
                all_values[original_batch_idx] = mb_values[j]
        
        return all_action_scores, all_values

    def run(self, num_episodes: int, temperature: float, is_eval_mode: bool) -> Tuple[List[Dict], List[EpisodeStats]]:
        """
        Run N parallel rollouts until each episode is complete, using micro-batching.
        """
        # --- Initialization ---
        envs = [TextWorldEnvironment(config=self.config, is_eval_env=is_eval_mode) for _ in range(num_episodes)]
        states = [env.reset() for env in envs]
        active_mask = [True] * num_episodes
        
        # Trackers
        current_rewards = np.zeros(num_episodes)
        current_lengths = np.zeros(num_episodes, dtype=int)
        
        # Output buffers
        rollout_buffer = []
        finished_episodes = []

        # Detailed logging setup for evaluation
        num_sample_games = min(num_episodes, 5) if is_eval_mode else 0
        game_logs = [[] for _ in range(num_sample_games)]
        if is_eval_mode:
            for i in range(num_sample_games):
                game_logs[i].append(f"=== SAMPLE GAME {i + 1} ===")
                game_logs[i].append(f"{states[i]}\n")

        with tqdm(total=num_episodes, desc="Completing parallel episodes") as pbar:
            while any(active_mask):
                active_indices = [i for i, active in enumerate(active_mask) if active]
                batch_states = [states[i] for i in active_indices]
                batch_actions = [envs[i].get_valid_actions() for i in active_indices]

                # --- CHANGE: Process the batch using the micro-batching helper ---
                action_scores, values = self._process_micro_batches(batch_states, batch_actions)

                # The rest of the loop remains the same, as it processes the aggregated results.
                for i, original_idx in enumerate(active_indices):
                    if not batch_actions[i]:
                        chosen_action, action_idx, old_logprob, done, info = "wait", 0, -np.inf, True, {}
                    else:
                        action_idx, old_logprob = self.temperature_sampling(action_scores[i], temperature)
                        chosen_action = batch_actions[i][action_idx]
                        next_state, reward, done, info = envs[original_idx].step(chosen_action)

                    # Store transition data for training buffer
                    if not is_eval_mode:
                        rollout_buffer.append({
                            "env_idx": original_idx, 
                            "state": states[original_idx], 
                            "action": chosen_action,
                            "action_idx": action_idx, 
                            "available_actions": batch_actions[i],
                            "old_logprob": old_logprob, 
                            "value": values[i],
                            "reward": reward, 
                            "done": done,
                        })

                    # Update trackers
                    current_rewards[original_idx] += reward
                    current_lengths[original_idx] += 1
                    states[original_idx] = next_state if not done else None
                    
                    # Detailed logging for sample eval games
                    if original_idx < num_sample_games:
                        action_probs = self._softmax(np.array(action_scores[i]))
                        log_entry = [
                            f"Step {current_lengths[original_idx]}:",
                            f"  Action Probabilities: {[f'{p:.3f}' for p in action_probs]}",
                            f"  Chosen Action: {chosen_action} | State Value: {values[i]:.3f}",
                            f"  Reward: {reward}",
                        ]
                        if not done: log_entry.append(f"\n{next_state}\n")
                        game_logs[original_idx].extend(log_entry)

                    # Handle finished episodes
                    if done or current_lengths[original_idx] >= self.config.num_steps:
                        active_mask[original_idx] = False
                        game_won = info.get("won", False)
                        
                        game_log_str = None
                        if original_idx < num_sample_games:
                            log_summary = [
                                f"=== GAME {original_idx + 1} SUMMARY ===",
                                f"Total Steps: {current_lengths[original_idx]}",
                                f"Total Reward: {current_rewards[original_idx]:.3f}",
                                f"Completed: {'Yes (Won)' if game_won else 'No (Truncated or Lost)'}",
                                "=" * 50
                            ]
                            game_logs[original_idx].extend(log_summary)
                            game_log_str = "\n".join(game_logs[original_idx])

                        finished_episodes.append(EpisodeStats(
                            length=current_lengths[original_idx],
                            reward=current_rewards[original_idx],
                            completed=game_won,
                            game_log=game_log_str
                        ))
                        pbar.update(1)
        
        for env in envs:
            env.close()

        return rollout_buffer, finished_episodes

    def temperature_sampling(self, raw_scores, temperature: float) -> Tuple[int, float]:
        """Apply temperature sampling to raw action scores."""
        scores = np.array(raw_scores)
        if not scores.any(): return 0, -np.inf

        # For greedy evaluation, set temp very low but non-zero to avoid division errors
        if temperature < 1e-5:
            action_idx = np.argmax(scores)
        else:
            scaled_scores = scores / temperature
            scaled_scores_stable = scaled_scores - np.max(scaled_scores)
            sampling_probs = np.exp(scaled_scores_stable) / np.sum(np.exp(scaled_scores_stable))
            action_idx = np.random.choice(len(sampling_probs), p=sampling_probs)
        
        original_scores_stable = scores - np.max(scores)
        exp_scores = np.exp(original_scores_stable)
        log_sum_exp = np.log(np.sum(exp_scores))
        original_logprobs = original_scores_stable - log_sum_exp
        old_logprob = original_logprobs[action_idx]
        
        return int(action_idx), old_logprob