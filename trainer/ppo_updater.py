import torch
import torch.nn.functional as F
import numpy as np
import wandb
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.amp import autocast
from tqdm import tqdm

from config import DynamicConfigManager


class PPOUpdater:
    """Handles PPO policy updates and advantage computation"""

    def __init__(self, config, policy, optimizer, scheduler, scaler, device, logger):
        self.config = config
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.logger = logger
        self.dynamic_config = DynamicConfigManager(config, logger)
        self.kl_coef = config.kl_coef if config.use_kl_penalty else 0.0

    def compute_advantages(self, rollout_buffer: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns. This version is simplified for episode-centric
        rollouts, where each environment's data is a single, complete episode.
        """
        # Group experiences by environment index, which now corresponds to a single episode.
        experiences_by_env = defaultdict(list)
        for exp in rollout_buffer:
            experiences_by_env[exp["env_idx"]].append(exp)

        # Buffers to hold the flat list of advantages and returns for the entire batch.
        advantages_flat = [0] * len(rollout_buffer)
        returns_flat = [0] * len(rollout_buffer)

        # Process each episode's trajectory independently.
        for env_idx in sorted(experiences_by_env.keys()):
            experiences = experiences_by_env[env_idx]
            
            # Since we collect full episodes, the value after the last state is always 0.
            # There is no need to bootstrap from the value function for truncated rollouts.
            next_value = 0.0
            last_advantage = 0.0
            
            # Loop backwards through the episode's experiences.
            for i, exp in reversed(list(enumerate(experiences))):
                delta = exp["reward"] + self.config.gamma * next_value - exp["value"]
                advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage
                
                # We need the original index from the flat rollout_buffer to place the results.
                # This is a bit tricky, so we'll just re-calculate the buffer later.
                # For now, let's build per-episode lists.
                
                # Store the computed advantage and return for this step.
                returns_flat[rollout_buffer.index(exp)] = advantage + exp["value"]
                advantages_flat[rollout_buffer.index(exp)] = advantage

                # The next_value for the previous step is the value of the current state.
                next_value = exp["value"]
                last_advantage = advantage

        advantages = np.array(advantages_flat, dtype=np.float32)
        returns = np.array(returns_flat, dtype=np.float32)

        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update(self, rollout_buffer: List[Dict], iteration: int, temperature: float):
        """PPO policy update"""
        self.logger.info(f"\nIteration: {iteration}")

        if self.config.dynamic_config:
            changes = self.dynamic_config.check_and_update() if iteration > 0 else None

            if changes:
                self.logger.info(f"DYNAMIC CONFIG UPDATED at iteration {iteration}")
                
                for param, (old_val, new_val) in changes.items():
                    self.logger.info(f"  {param}: {old_val} → {new_val}")
                    if param == 'learning_rate':
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_val
                        self.logger.info(f"  Learning rate applied to optimizer")
            
        advantages, returns = self.compute_advantages(rollout_buffer)

        self.log_value_function_diagnostics(rollout_buffer, iteration)

        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_logprobs = torch.FloatTensor(
            [exp["old_logprob"] for exp in rollout_buffer]
        ).to(self.device)

        # ADVANTAGE TRACKING: Compute advantage statistics
        advantage_stats = {
            "mean": advantages.mean().item(),
            "std": advantages.std().item(),
            "min": advantages.min().item(),
            "max": advantages.max().item(),
            "median": advantages.median().item(),
            "q25": advantages.quantile(0.25).item(),
            "q75": advantages.quantile(0.75).item(),
            "positive_ratio": (advantages > 0).float().mean().item(),
            "zero_ratio": (advantages == 0).float().mean().item(),
            "abs_mean": advantages.abs().mean().item()
        }

        # Check for non-finite values in inputs
        if not (returns.isfinite().all() and old_logprobs.isfinite().all()):
            self.logger.error(
                "Non-finite values in returns or old_logprobs. Aborting update."
            )
            wandb.log({"error": "Non-finite values in returns or old_logprobs"})
            return

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_combined_loss = 0
        num_updates = 0
        max_seq_len = 0

        advantage_usage_stats = []

        accumulation_steps = self.config.accumulation_steps
        effective_batch_size = min(self.config.batch_size, len(rollout_buffer))

        # PPO epochs
        for epoch in tqdm(range(self.config.ppo_epochs), desc="Running PPO"):
            indices = torch.randperm(len(rollout_buffer))

            epoch_kl_values = []

            for start in range(0, len(rollout_buffer), effective_batch_size):
                end = start + effective_batch_size
                batch_indices = indices[start:end]

                # Get batch data
                batch_experiences = [rollout_buffer[i] for i in batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]

                # ADVANTAGE TRACKING: Track batch advantage statistics
                batch_adv_stats = {
                    "batch_adv_mean": batch_advantages.mean().item(),
                    "batch_adv_std": batch_advantages.std().item(),
                    "batch_adv_abs_mean": batch_advantages.abs().mean().item(),
                }
                advantage_usage_stats.append(batch_adv_stats)

                # Prepare inputs
                batch_states = [exp["state"] for exp in batch_experiences]

                batch_action_lists = [exp["available_actions"] for exp in batch_experiences]
                batch_action_indices = [exp["action_idx"] for exp in batch_experiences]

                # Forward pass
                with autocast(self.device.type):
                    current_action_logprobs, current_values, current_seq_len = self.policy.evaluate_actions(
                        batch_states, batch_action_lists, temperature
                    )

                    max_seq_len = max(max_seq_len, current_seq_len)

                    # Compute KL divergence if enabled
                    kl_div = torch.tensor(0.0, device=self.device)
                    if self.config.use_kl_penalty:
                        # --- PERMANENT SANITY CHECK ---
                        # On the first batch of the first iteration, verify that the KL
                        # between the two models in eval mode is zero. This prevents regressions.
                        if iteration == 0 and epoch == 0 and start == 0:
                            self.policy.eval()
                            with torch.no_grad():
                                eval_logprobs, _, _ = self.policy.evaluate_actions(
                                    batch_states, batch_action_lists, temperature
                                )
                                ref_logprobs = self.policy.get_reference_action_scores(
                                    batch_states, batch_action_lists, temperature
                                )
                                initial_kl = self.policy.compute_kl_divergence(
                                    eval_logprobs, ref_logprobs
                                )
                            # Allow for a tiny tolerance for floating point artifacts
                            assert torch.allclose(initial_kl, torch.tensor(0.0), atol=1e-3), \
                                f"Initial KL divergence in eval mode is not zero: {initial_kl.item()}"
                            self.logger.info("✅ Initial KL divergence sanity check passed.")
                            self.policy.train() # Restore train mode
                        # --- END OF SANITY CHECK ---

                        # For the actual loss, we calculate KL between the stochastic (train)
                        # policy and the deterministic (eval) reference policy.
                        reference_action_logprobs = self.policy.get_reference_action_scores(
                            batch_states, batch_action_lists, temperature
                        )
                        
                        # Compute KL divergence
                        kl_div = self.policy.compute_kl_divergence(
                            current_action_logprobs,
                            reference_action_logprobs
                        )
                        
                        epoch_kl_values.append(kl_div.item())

                    batch_chosen_logprobs = []
                    batch_entropy = []
                    for i, logprobs in enumerate(current_action_logprobs):
                        probs_dist = torch.exp(logprobs)

                        # A) Extract the log_prob of the action that was chosen
                        action_idx = batch_action_indices[i]
                        batch_chosen_logprobs.append(logprobs[action_idx])

                        # B) Compute categorical entropy: H(p) = -sum(p * log(p))
                        entropy_dist = -torch.sum(probs_dist * logprobs)
                        batch_entropy.append(entropy_dist)

                    current_logprobs = torch.stack(batch_chosen_logprobs)
                    entropy = torch.stack(batch_entropy).mean()

                    # Compute PPO loss
                    current_values = current_values.view(-1)
                    
                    log_prob_diff = current_logprobs - batch_old_logprobs
                    ratio = torch.exp(torch.clamp(log_prob_diff, min=-5, max=5))

                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1 - self.config.epsilon_clip,
                            1 + self.config.epsilon_clip,
                        )
                        * batch_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(current_values, batch_returns)

                    if not (policy_loss.isfinite() and value_loss.isfinite() and entropy.isfinite()):
                        self.logger.warning("Non-finite loss values. Skipping batch.")
                        wandb.log({"warning": "Non-finite loss values"})
                        continue

                    # regularizer_loss = F.softplus(-(torch.cat(current_action_logprobs) + 12)).sum()

                    total_loss = (
                        policy_loss
                        + self.dynamic_config.get('value_loss_coef') * value_loss
                        - self.dynamic_config.get('entropy_coef') * entropy
                        # + 0.01 * regularizer_loss
                        + self.kl_coef * kl_div
                    ) / accumulation_steps

                self.scaler.scale(total_loss).backward()

                if ((start + effective_batch_size) // effective_batch_size) % accumulation_steps == 0 or end >= len(rollout_buffer):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm,
                        error_if_nonfinite=False
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                total_combined_loss += total_loss.item() * accumulation_steps
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

                del (
                    current_logprobs,
                    entropy,
                    current_values,
                    ratio,
                    surr1,
                    surr2,
                    policy_loss,
                    value_loss,
                    total_loss,
                )

        self.scheduler.step()

        if advantage_usage_stats:
            avg_batch_adv_mean = sum(stat["batch_adv_mean"] for stat in advantage_usage_stats) / len(advantage_usage_stats)
            avg_batch_adv_std = sum(stat["batch_adv_std"] for stat in advantage_usage_stats) / len(advantage_usage_stats)
            avg_batch_adv_abs_mean = sum(stat["batch_adv_abs_mean"] for stat in advantage_usage_stats) / len(advantage_usage_stats)
        else:
            avg_batch_adv_mean = avg_batch_adv_std = avg_batch_adv_abs_mean = 0.0

        if num_updates > 0:
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "iteration": iteration,
                    "policy_loss": total_policy_loss / num_updates,
                    "value_loss": total_value_loss / num_updates,
                    "total_loss": total_combined_loss / num_updates,
                    "entropy": total_entropy / num_updates,
                    "learning_rate": current_lr,
                    "kl_divergence": np.mean(epoch_kl_values) if epoch_kl_values else 0.0,
                    "max_seq_length": max_seq_len,
                    "dynamic/value_loss_coef": self.dynamic_config.get('value_loss_coef'),
                    "dynamic/entropy_coef": self.dynamic_config.get('entropy_coef'),
                    "advantages/mean": advantage_stats["mean"],
                    "advantages/std": advantage_stats["std"],
                    "advantages/min": advantage_stats["min"],
                    "advantages/max": advantage_stats["max"],
                    "advantages/median": advantage_stats["median"],
                    "advantages/q25": advantage_stats["q25"],
                    "advantages/q75": advantage_stats["q75"],
                    "advantages/positive_ratio": advantage_stats["positive_ratio"],
                    "advantages/zero_ratio": advantage_stats["zero_ratio"],
                    "advantages/abs_mean": advantage_stats["abs_mean"],
                    "advantages/batch_avg_mean": avg_batch_adv_mean,
                    "advantages/batch_avg_std": avg_batch_adv_std,
                    "advantages/batch_avg_abs_mean": avg_batch_adv_abs_mean,
                }
            )

            self.logger.info(
                f"Policy Loss: {total_policy_loss/num_updates:.4f}, "
                f"Value Loss: {total_value_loss/num_updates:.4f}, "
                f"Entropy: {total_entropy/num_updates:.4f}, "
                f"LR: {current_lr:.2e}, "
                f"KL Stats - Mean: {np.mean(epoch_kl_values):.4f}, "
                f"Max: {np.max(epoch_kl_values, initial=0.0):.4f}, "
                f"Min: {np.min(epoch_kl_values, initial=0.0):.4f}, "
                f"Max Batch Tokens: {max_seq_len}"
                "\n\n"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _has_any_finite_tensor(self, nested_tensor_list):
        """Check if any tensor in nested list has finite values"""
        for sublist in nested_tensor_list:
            for tensor in sublist:
                if torch.isfinite(tensor).all():
                    return True
        return False

    def compute_td_errors_and_trajectories(self, rollout_buffer: List[Dict]) -> Dict:
            """Compute TD errors and extract value trajectories for analysis."""
            # This correctly returns a list of episode lists.
            episodes = self._reconstruct_episodes_from_buffer(rollout_buffer)
            
            all_td_errors = []
            value_trajectories = []
            episode_outcomes = []

            # --- CORRECTED LOOP ---
            # Iterate directly over the list of episodes.
            for experiences in episodes:
                if not experiences:
                    continue

                # Get env_idx from the first experience in the episode.
                env_idx = experiences[0]["env_idx"]
                
                episode_td_errors = []
                episode_values = []
                episode_rewards = []
                
                for i, exp in enumerate(experiences):
                    current_value = exp["value"]
                    reward = exp["reward"]
                    
                    # Compute TD error: δ = r + γV(s') - V(s)
                    if i < len(experiences) - 1:
                        next_value = experiences[i + 1]["value"]
                        td_error = reward + self.config.gamma * next_value - current_value
                    else:
                        # For the last step, the next_value depends on whether it was a true 'done' state.
                        next_value = 0.0 if exp["done"] else exp["value"] # Bootstrap if truncated
                        td_error = reward + self.config.gamma * next_value - current_value

                    episode_td_errors.append(td_error)
                    episode_values.append(current_value)
                    episode_rewards.append(reward)
                
                all_td_errors.extend(episode_td_errors)
                
                # Determine episode outcome
                # Check if any step in the episode had a positive reward (a more robust way to check for a win)
                episode_won = any(exp['reward'] > 0 for exp in experiences)
                
                value_trajectories.append({
                    "env_idx": env_idx,
                    "values": episode_values,
                    "rewards": episode_rewards,
                    "td_errors": episode_td_errors,
                    "won": episode_won,
                    "episode_length": len(experiences)
                })
                episode_outcomes.append(episode_won)
            
            td_errors_array = np.array(all_td_errors) if all_td_errors else np.array([0.0])
            
            return {
                "td_errors": td_errors_array,
                "td_error_mean": td_errors_array.mean(),
                "td_error_std": td_errors_array.std(),
                "td_error_abs_mean": np.abs(td_errors_array).mean(),
                "value_trajectories": value_trajectories,
                "win_rate": np.mean(episode_outcomes) if episode_outcomes else 0.0,
                "num_episodes": len(value_trajectories)
            }
    
    def _reconstruct_episodes_from_buffer(self, rollout_buffer: List[Dict]) -> List[List[Dict]]:
        """Reconstructs individual episodes from a flat rollout buffer."""
        episodes = []
        current_episodes = defaultdict(list)
        
        # Group by environment index first
        experiences_by_env = defaultdict(list)
        for exp in rollout_buffer:
            experiences_by_env[exp["env_idx"]].append(exp)

        for env_idx in sorted(experiences_by_env.keys()):
            for exp in experiences_by_env[env_idx]:
                current_episodes[env_idx].append(exp)
                if exp["done"]:
                    episodes.append(current_episodes[env_idx])
                    current_episodes[env_idx] = []
        
        # Add any unfinished episodes at the end of the buffer
        for env_idx in sorted(current_episodes.keys()):
            if current_episodes[env_idx]:
                episodes.append(current_episodes[env_idx])

        return episodes

    def log_value_function_diagnostics(self, rollout_buffer: List[Dict], iteration: int):
        """Log comprehensive value function diagnostics."""
        diagnostics = self.compute_td_errors_and_trajectories(rollout_buffer)
        
        # Log basic TD error metrics
        wandb.log({
            "value_diagnostics/td_error_mean": diagnostics["td_error_mean"],
            "value_diagnostics/td_error_std": diagnostics["td_error_std"],
            "value_diagnostics/td_error_abs_mean": diagnostics["td_error_abs_mean"],
            "value_diagnostics/win_rate": diagnostics["win_rate"],
            "iteration": iteration
        })
        
        # Log value trajectory analysis
        trajectories = diagnostics["value_trajectories"]
        
        if trajectories:
            # Separate winning and losing episodes
            winning_episodes = [t for t in trajectories if t["won"]]
            losing_episodes = [t for t in trajectories if not t["won"]]
            
            if winning_episodes:
                win_values = [np.mean(t["values"]) for t in winning_episodes]
                win_initial_values = [t["values"][0] for t in winning_episodes]
                win_final_values = [t["values"][-1] for t in winning_episodes]
                
                wandb.log({
                    "value_diagnostics/winning_episodes_mean_value": np.mean(win_values),
                    "value_diagnostics/winning_episodes_initial_value": np.mean(win_initial_values),
                    "value_diagnostics/winning_episodes_final_value": np.mean(win_final_values),
                    "iteration": iteration
                })
            
            if losing_episodes:
                loss_values = [np.mean(t["values"]) for t in losing_episodes]
                loss_initial_values = [t["values"][0] for t in losing_episodes]
                loss_final_values = [t["values"][-1] for t in losing_episodes]
                
                wandb.log({
                    "value_diagnostics/losing_episodes_mean_value": np.mean(loss_values),
                    "value_diagnostics/losing_episodes_initial_value": np.mean(loss_initial_values),
                    "value_diagnostics/losing_episodes_final_value": np.mean(loss_final_values),
                    "iteration": iteration
                })
        
        # Log detailed trajectory plots every N iterations
        if iteration % 5 == 0 and trajectories:
            self._plot_value_trajectories(trajectories, iteration)
        
        self.logger.info(
            f"Value Diagnostics - TD Error: {diagnostics['td_error_abs_mean']:.4f}, "
            f"Win Rate: {diagnostics['win_rate']:.2%}, "
            f"Episodes: {diagnostics['num_episodes']}"
        )

    def _plot_value_trajectories(self, trajectories: List[Dict], iteration: int):
        """Create plots of value trajectories for winning vs losing episodes."""
        import matplotlib.pyplot as plt
        
        winning_trajectories = [t for t in trajectories if t["won"]]
        losing_trajectories = [t for t in trajectories if not t["won"]]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Value Function Analysis - Iteration {iteration}')
        
        # Plot 1: Value trajectories over time
        ax1 = axes[0, 0]
        for i, traj in enumerate(winning_trajectories[:5]):  # Plot first 5 winning episodes
            ax1.plot(traj["values"], 'g-', alpha=0.7, label='Winning' if i == 0 else "")
        for i, traj in enumerate(losing_trajectories[:5]):   # Plot first 5 losing episodes
            ax1.plot(traj["values"], 'r-', alpha=0.7, label='Losing' if i == 0 else "")
        ax1.set_title('Value Trajectories')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: TD errors over time
        ax2 = axes[0, 1]
        for i, traj in enumerate(winning_trajectories[:5]):
            ax2.plot(traj["td_errors"], 'g-', alpha=0.7, label='Winning' if i == 0 else "")
        for i, traj in enumerate(losing_trajectories[:5]):
            ax2.plot(traj["td_errors"], 'r-', alpha=0.7, label='Losing' if i == 0 else "")
        ax2.set_title('TD Errors')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('TD Error')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Value distribution comparison
        ax3 = axes[1, 0]
        if winning_trajectories:
            win_values = [v for traj in winning_trajectories for v in traj["values"]]
            ax3.hist(win_values, bins=30, alpha=0.7, color='green', label='Winning', density=True)
        if losing_trajectories:
            loss_values = [v for traj in losing_trajectories for v in traj["values"]]
            ax3.hist(loss_values, bins=30, alpha=0.7, color='red', label='Losing', density=True)
        ax3.set_title('Value Distribution')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Initial vs Final values
        ax4 = axes[1, 1]
        if winning_trajectories:
            win_initial = [t["values"][0] for t in winning_trajectories]
            win_final = [t["values"][-1] for t in winning_trajectories]
            ax4.scatter(win_initial, win_final, c='green', alpha=0.7, label='Winning')
        if losing_trajectories:
            loss_initial = [t["values"][0] for t in losing_trajectories]
            loss_final = [t["values"][-1] for t in losing_trajectories]
            ax4.scatter(loss_initial, loss_final, c='red', alpha=0.7, label='Losing')
        ax4.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)  # Diagonal line
        ax4.set_title('Initial vs Final Values')
        ax4.set_xlabel('Initial Value')
        ax4.set_ylabel('Final Value')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"value_trajectories_plot": wandb.Image(fig)})
        plt.close()