import torch
import torch.nn.functional as F
import numpy as np
import wandb
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.amp import autocast
from tqdm import tqdm

from config.dynamic_config_manager import DynamicConfigManager


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
        self.cumulative_episodes = 0

        self.dynamic_config = DynamicConfigManager(config, logger)

        # Track when coefficients change for logging
        self.last_config_check = 0
        self.config_check_interval = 1  # Check every batch

    def compute_advantages(
        self, rollout_buffer: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        grouped_experiences = defaultdict(list)
        for exp in rollout_buffer:
            grouped_experiences[(exp["env_idx"], exp["episode_id"])].append(exp)

        all_advantages = []
        all_returns = []

        for key, experiences in grouped_experiences.items():
            if not experiences:
                continue

            last_advantage = 0.0
            # Determine the bootstrap value for the very last state in the trajectory.
            # If the episode was truncated (not 'done'), bootstrap from its value estimate.
            # Otherwise (it was a terminal state), the value is 0.
            last_exp = experiences[-1]
            if last_exp["truncated"] and not last_exp["done"]:
                # This handles episodes cut short by the rollout limit.
                next_value = last_exp["value"]
            else:
                # This handles true terminal states.
                next_value = 0.0

            advantages_ep = []
            returns_ep = []
            
            # Loop backwards through the experiences
            for exp in reversed(experiences):
                # The value of the state after the current one is 'next_value'.
                delta = exp["reward"] + self.config.gamma * next_value - exp["value"]
                
                # Calculate the advantage using the GAE formula
                advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage
                
                # The return is the advantage plus the value of the current state
                returns_ep.append(advantage + exp["value"])
                advantages_ep.append(advantage)

                # Update carriers for the next iteration (which is the previous time step)
                last_advantage = advantage
                next_value = exp["value"] # The current state's value becomes the next state's value for the previous step

            # Reverse the lists to be in chronological order and add to the main buffer
            advantages_ep.reverse()
            returns_ep.reverse()
            all_advantages.extend(advantages_ep)
            all_returns.extend(returns_ep)

        advantages = np.array(all_advantages, dtype=np.float32)
        returns = np.array(all_returns, dtype=np.float32)
        
        if len(advantages) > 1 and self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages = np.clip(advantages, -5, 5)

        return advantages, returns

    def ppo_update(self, rollout_buffer: List[Dict], iteration: int):
        """PPO policy update"""
        self.logger.info(f"\nIteration: {iteration}")

        if self.config.dynamic_config:
            changes = self.dynamic_config.check_and_update() if iteration > 0 else None

            if changes:
                self.logger.info(f"DYNAMIC CONFIG UPDATED at iteration {iteration}")
                
                for param, (old_val, new_val) in changes.items():
                    # Log to terminal for ALL parameters
                    self.logger.info(f"  {param}: {old_val} → {new_val}")

                    # Special handling for learning rate changes
                    if param == 'learning_rate':
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_val
                        self.logger.info(f"  Learning rate applied to optimizer")
            


        # Count unique episodes in the rollout buffer
        unique_episodes = len(
            set((exp["env_idx"], exp["episode_id"]) for exp in rollout_buffer)
        )
        self.cumulative_episodes += unique_episodes
        advantages, returns = self.compute_advantages(rollout_buffer)

        # self.log_value_function_diagnostics(rollout_buffer, iteration)

        # Existing validation
        # ranking_results = self.validate_value_function_ranking(rollout_buffer)
        # wandb.log({
        #     "value_diagnostics/ranking_accuracy": ranking_results["ranking_accuracy"],
        #     "iteration": iteration
        # })

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

        advantage_usage_stats = []

        accumulation_steps = self.config.accumulation_steps
        effective_batch_size = min(self.config.batch_size, len(rollout_buffer))

        # PPO epochs
        for epoch in tqdm(range(self.config.ppo_epochs), desc="Running PPO"):
            indices = torch.randperm(len(rollout_buffer))

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
                batch_action_lists = [
                    exp["available_actions"] for exp in batch_experiences
                ]
                batch_action_indices = [exp["action_idx"] for exp in batch_experiences]

                # Forward pass
                with autocast(self.device.type):
                    current_action_logprobs, current_values = self.policy.evaluate_actions(
                        batch_states, batch_action_lists
                    )

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
                    log_prob_diff = torch.clamp(log_prob_diff, min=-5, max=5)

                    ratio = torch.exp(log_prob_diff)
                    
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

                    regularizer_loss = F.softplus(-(torch.cat(current_action_logprobs) + 12)).sum()

                    total_loss = (
                        policy_loss
                        + self.dynamic_config.get('value_loss_coef') * value_loss
                        # - self.dynamic_config.get('entropy_coef') * entropy
                        + 0.01 * regularizer_loss
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
                    "cumulative_episodes": self.cumulative_episodes,
                    "episodes_per_update": unique_episodes,
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
                f"LR: {current_lr:.2e}"
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
        grouped_experiences = defaultdict(list)
        for exp in rollout_buffer:
            grouped_experiences[(exp["env_idx"], exp["episode_id"])].append(exp)
        
        all_td_errors = []
        value_trajectories = []
        episode_outcomes = []  # Track win/loss for each episode
        
        for (env_idx, episode_id), experiences in grouped_experiences.items():
            if not experiences:
                continue
                
            # Sort by step to ensure proper order
            experiences.sort(key=lambda x: x.get('step', 0))
            
            episode_td_errors = []
            episode_values = []
            episode_rewards = []
            episode_states = []
            
            for i, exp in enumerate(experiences):
                current_value = exp["value"]
                reward = exp["reward"]
                
                # Compute TD error: δ = r + γV(s') - V(s)
                if i < len(experiences) - 1:
                    next_value = experiences[i + 1]["value"]
                    td_error = reward + self.config.gamma * next_value - current_value
                else:
                    # Terminal state
                    if exp["done"] and not exp["truncated"]:
                        # True terminal state, next value is 0
                        td_error = reward - current_value
                    else:
                        # Truncated episode, use bootstrap value
                        td_error = reward + self.config.gamma * current_value - current_value
                
                episode_td_errors.append(td_error)
                episode_values.append(current_value)
                episode_rewards.append(reward)
                
                # Store state info for analysis (you might want to truncate long states)
                state_preview = exp["state"][:100] + "..." if len(exp["state"]) > 100 else exp["state"]
                episode_states.append(state_preview)
            
            all_td_errors.extend(episode_td_errors)
            
            # Determine episode outcome (you'll need to adapt this to your reward structure)
            final_reward = experiences[-1]["reward"]
            episode_won = final_reward > 0  # Adjust this condition based on your game
            
            value_trajectories.append({
                "env_idx": env_idx,
                "episode_id": episode_id,
                "values": episode_values,
                "rewards": episode_rewards,
                "td_errors": episode_td_errors,
                "states": episode_states,
                "won": episode_won,
                "final_reward": final_reward,
                "episode_length": len(experiences)
            })
            episode_outcomes.append(episode_won)
        
        td_errors_array = np.array(all_td_errors)
        
        return {
            "td_errors": td_errors_array,
            "td_error_mean": td_errors_array.mean(),
            "td_error_std": td_errors_array.std(),
            "td_error_abs_mean": np.abs(td_errors_array).mean(),
            "value_trajectories": value_trajectories,
            "win_rate": np.mean(episode_outcomes) if episode_outcomes else 0.0,
            "num_episodes": len(value_trajectories)
        }

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

    def validate_value_function_ranking(self, rollout_buffer: List[Dict]) -> Dict:
        """Check if value function correctly ranks obviously good vs bad states."""
        diagnostics = self.compute_td_errors_and_trajectories(rollout_buffer)
        trajectories = diagnostics["value_trajectories"]
        
        ranking_issues = []
        correct_rankings = 0
        total_comparisons = 0
        
        # Compare winning vs losing episodes
        winning_episodes = [t for t in trajectories if t["won"]]
        losing_episodes = [t for t in trajectories if not t["won"]]
        
        for win_ep in winning_episodes:
            for loss_ep in losing_episodes:
                total_comparisons += 1
                win_avg_value = np.mean(win_ep["values"])
                loss_avg_value = np.mean(loss_ep["values"])
                
                if win_avg_value > loss_avg_value:
                    correct_rankings += 1
                else:
                    ranking_issues.append({
                        "win_episode": win_ep["episode_id"],
                        "win_avg_value": win_avg_value,
                        "loss_episode": loss_ep["episode_id"], 
                        "loss_avg_value": loss_avg_value,
                        "win_final_reward": win_ep["final_reward"],
                        "loss_final_reward": loss_ep["final_reward"]
                    })
        
        ranking_accuracy = correct_rankings / total_comparisons if total_comparisons > 0 else 0.0
        
        return {
            "ranking_accuracy": ranking_accuracy,
            "correct_rankings": correct_rankings,
            "total_comparisons": total_comparisons,
            "ranking_issues": ranking_issues[:5]  # Show first 5 issues for debugging
        }