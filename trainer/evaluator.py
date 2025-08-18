import torch
import numpy as np
import wandb
from typing import Tuple, List, Dict
from tqdm import tqdm
from torch.amp import autocast

from env import TextWorldEnvironment


class Evaluator:
    """
    Handles policy evaluation by running a fixed set of N environments in parallel.
    When an environment finishes, it becomes inactive and is not reset. The total
    evaluation time is determined by the longest-running episode.
    """

    def __init__(self, policy, config, device, logger):
        self.policy = policy
        self.config = config
        self.device = device
        self.logger = logger
        self.num_envs = config.num_evals

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for x with numerical stability."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def run_evaluation(self, iteration: int) -> Tuple[float, float]:
        """
        Run N parallel evaluation rollouts, where N is the number of environments.

        Args:
            iteration (int): The current training iteration number, for logging purposes.

        Returns:
            A tuple containing the average episode length and average episode reward.
        """
        self.logger.info(f"\n\nRunning fixed-set parallel evaluation with {self.num_envs} environments...")

        # We will log up to 5 games in detail, or fewer if num_envs is smaller.
        num_sample_games = min(self.num_envs, 5)

        # --- Initialization ---
        envs = [TextWorldEnvironment(config=self.config, is_eval_env=True) for _ in range(self.num_envs)]
        states = [env.reset() for env in envs]

        # A mask to track which environments are still running.
        active_mask = [True] * self.num_envs

        # Trackers for each parallel environment.
        current_rewards = np.zeros(self.num_envs)
        current_lengths = np.zeros(self.num_envs, dtype=int)
        
        # Buffers to store detailed logs for the sample games.
        game_logs = [[] for _ in range(num_sample_games)]
        for i in range(num_sample_games):
            game_logs[i].append(f"=== SAMPLE GAME {i + 1} ===")
            game_logs[i].append(f"{states[i]}\n")

        finished_episodes = []

        # --- Main Evaluation Loop ---
        # The loop continues as long as at least one environment is active.
        with tqdm(total=self.num_envs, desc="Completing parallel episodes") as pbar:
            while any(active_mask):
                # 1. Build a batch ONLY from active environments.
                active_indices = [i for i, active in enumerate(active_mask) if active]
                batch_states = [states[i] for i in active_indices]
                batch_actions = [envs[i].get_valid_actions() for i in active_indices]

                # 2. Get model outputs for the active batch. The batch size may shrink over time.
                with torch.no_grad(), autocast(self.device.type):
                    action_scores, values = self.policy.evaluate_for_rollout(batch_states, batch_actions)

                # 3. Process results and step each active environment.
                for i, original_idx in enumerate(active_indices):
                    # If an environment has no actions, it's a terminal state.
                    if not batch_actions[i]:
                        done = True
                        info = {} # Ensure info dict exists
                    else:
                        # scores = np.array(action_scores[i])
                        # action_idx = np.argmax(scores)
                        scores = action_scores[i]
                        action_idx, _ = self.temperature_sampling(scores, temperature=1.0) 
                        chosen_action = batch_actions[i][action_idx]

                        # Log details for sample games if this is a logging-enabled slot.
                        if original_idx < num_sample_games:
                            action_probs = self._softmax(scores)
                            game_logs[original_idx].append(f"Step {current_lengths[original_idx] + 1}:")
                            game_logs[original_idx].append(f"  Action Probabilities: {[f'{p:.3f}' for p in action_probs]}")
                            game_logs[original_idx].append(f"  Chosen Action: {chosen_action} | State Value: {values[i]:.3f}")

                        next_state, reward, done, info = envs[original_idx].step(chosen_action)
                        
                        current_rewards[original_idx] += reward
                        current_lengths[original_idx] += 1
                        states[original_idx] = next_state
                        
                        if original_idx < num_sample_games:
                            game_logs[original_idx].append(f"  Reward: {reward}")
                            if not done:
                                game_logs[original_idx].append(f"\n{next_state}\n")

                    # 4. Handle finished episodes.
                    if done or current_lengths[original_idx] >= self.config.num_steps:
                        # Mark this environment as inactive for the next loop.
                        active_mask[original_idx] = False

                        game_won = info.get("won", False)

                        # Finalize the detailed log if this was a sample game.
                        if original_idx < num_sample_games:
                            game_logs[original_idx].append(f"=== GAME {original_idx + 1} SUMMARY ===")
                            game_logs[original_idx].append(f"Total Steps: {current_lengths[original_idx]}")
                            game_logs[original_idx].append(f"Total Reward: {current_rewards[original_idx]:.3f}")
                            completion_status = "Yes (Won)" if game_won else "No (Truncated or Lost)"
                            game_logs[original_idx].append(f"Completed: {completion_status}")
                            game_logs[original_idx].append("=" * 50)

                        # Store the final result.
                        finished_episodes.append({
                            "length": current_lengths[original_idx],
                            "reward": current_rewards[original_idx],
                            "completed": game_won,
                            "game_log": "\n".join(game_logs[original_idx]) if original_idx < num_sample_games else None
                        })
                        pbar.update(1)
                        # --- NO RESET HERE. The environment slot is now done permanently. ---
        
        # --- Cleanup and Reporting ---
        for env in envs:
            env.close()

        avg_episode_length = np.mean([ep["length"] for ep in finished_episodes])
        avg_episode_reward = np.mean([ep["reward"] for ep in finished_episodes])
        completed_count = sum(1 for ep in finished_episodes if ep["completed"])
        
        sample_games_logs = sorted([ep["game_log"] for ep in finished_episodes if ep["game_log"] is not None])
        sample_completed = sum(1 for ep in finished_episodes[:num_sample_games] if ep["completed"])

        self.logger.info(
            f"Evaluation completed: {len(finished_episodes)} total episodes run, "
            f"{completed_count} won, "
            f"Avg Length: {avg_episode_length:.2f}, "
            f"Avg Reward: {avg_episode_reward:.4f}"
        )

        # Log sample games to file.
        all_games_text = "\n\n".join(sample_games_logs)
        with open(f"evaluations/{wandb.run.name}.txt", "a") as f:
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Evaluation Metrics - Total Episodes: {len(finished_episodes)}, ")
            f.write(f"Completed (Won): {completed_count}, ")
            f.write(f"Sample Games Won: {sample_completed}/{num_sample_games}, ")
            f.write(f"Avg Length: {avg_episode_length:.2f}, Avg Reward: {avg_episode_reward:.4f}\n\n")
            f.write(all_games_text)
            f.write("\n\n")

        torch.cuda.empty_cache()

        return avg_episode_length, avg_episode_reward
    
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