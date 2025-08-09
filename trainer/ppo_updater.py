import torch
import torch.nn.functional as F
import numpy as np
import wandb
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.amp import autocast
from tqdm import tqdm


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

    def compute_advantages(
        self, rollout_buffer: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns with episode-level diagnostics."""
        grouped_experiences = defaultdict(list)
        for exp in rollout_buffer:
            grouped_experiences[(exp["env_idx"], exp["episode_id"])].append(exp)

        all_advantages = []
        all_returns = []

        for key, experiences in grouped_experiences.items():
            if not experiences:
                continue

            last_exp = experiences[-1]
            last_value = last_exp["value"] if last_exp["truncated"] else 0.0
            last_advantage = 0.0

            advantages_ep = []
            returns_ep = []
            
            for exp in reversed(experiences):
                if exp["done"]:
                    delta = exp["reward"] - exp["value"]
                    last_value = 0.0
                else:
                    delta = (
                        exp["reward"] + self.config.gamma * last_value - exp["value"]
                    )

                advantage = (
                    delta + self.config.gamma * self.config.gae_lambda * last_advantage
                )
                returns_ep.append(advantage + exp["value"])
                advantages_ep.append(advantage)

                last_value = exp["value"]
                last_advantage = advantage

            advantages_ep.reverse()
            returns_ep.reverse()

            all_advantages.extend(advantages_ep)
            all_returns.extend(returns_ep)

        advantages = np.array(all_advantages, dtype=np.float32)
        returns = np.array(all_returns, dtype=np.float32)
        
        if len(advantages) > 1 and self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optional clipping after normalization
        advantages = np.clip(advantages, -5, 5)

        return advantages, returns

    def ppo_update(self, rollout_buffer: List[Dict], iteration: int):
        """PPO policy update"""
        # Count unique episodes in the rollout buffer
        unique_episodes = len(
            set((exp["env_idx"], exp["episode_id"]) for exp in rollout_buffer)
        )
        self.cumulative_episodes += unique_episodes
        advantages, returns = self.compute_advantages(rollout_buffer)

        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_logprobs = torch.FloatTensor(
            [exp["old_logprob"] for exp in rollout_buffer]
        ).to(self.device)

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
        num_updates = 0

        accumulation_steps = self.config.accumulation_steps
        effective_batch_size = min(self.config.batch_size, len(rollout_buffer))

        # PPO epochs
        for epoch in tqdm(range(self.config.ppo_epochs), desc="Running PPO"):
            indices = torch.randperm(len(rollout_buffer))

            batch_count = 0
            for start in range(0, len(rollout_buffer), effective_batch_size):
                end = start + effective_batch_size
                batch_indices = indices[start:end]

                # Get batch data
                batch_experiences = [rollout_buffer[i] for i in batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]

                # Prepare inputs
                batch_states = [exp["state"] for exp in batch_experiences]
                batch_action_lists = [
                    exp["available_actions"] for exp in batch_experiences
                ]
                batch_action_indices = [exp["action_idx"] for exp in batch_experiences]

                # Forward pass
                with autocast(self.device.type):
                    current_action_logits, current_values = self.policy.evaluate_actions(
                        batch_states, batch_action_lists
                    )

                    batch_chosen_logprobs = []
                    batch_entropy = []
                    for i, logits in enumerate(current_action_logits):
                        if logits.nelement() == 0:
                            # Handle cases with no available actions if they can occur
                            continue
                        
                        # Use log_softmax for numerical stability. This creates a true
                        # log probability distribution over the available actions.
                        log_probs_dist = F.log_softmax(logits, dim=-1)
                        probs_dist = torch.exp(log_probs_dist)

                        # A) Extract the log_prob of the action that was chosen
                        action_idx = batch_action_indices[i]
                        batch_chosen_logprobs.append(log_probs_dist[action_idx])

                        # B) Compute categorical entropy: H(p) = -sum(p * log(p))
                        entropy_dist = -torch.sum(probs_dist * log_probs_dist)
                        batch_entropy.append(entropy_dist)

                    if not batch_chosen_logprobs:
                        self.logger.warning("Skipping batch with no valid actions.")
                        continue

                    current_logprobs = torch.stack(batch_chosen_logprobs)
                    entropy = torch.stack(batch_entropy).mean()

                    # Compute PPO loss
                    current_values = current_values.view(-1)
                    
                    ratio = torch.exp(current_logprobs - batch_old_logprobs)
                    
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

                    total_loss = (
                        policy_loss
                        + self.config.value_loss_coef * value_loss
                        - self.config.entropy_coef * entropy
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

        if num_updates > 0:
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "iteration": iteration,
                    "policy_loss": total_policy_loss / num_updates,
                    "value_loss": total_value_loss / num_updates,
                    "entropy": total_entropy / num_updates,
                    "learning_rate": current_lr,
                    "cumulative_episodes": self.cumulative_episodes,
                    "episodes_per_update": unique_episodes,
                }
            )

            self.logger.info(
                f"Policy Loss: {total_policy_loss/num_updates:.4f}, "
                f"Value Loss: {total_value_loss/num_updates:.4f}, "
                f"Entropy: {total_entropy/num_updates:.4f}, "
                f"LR: {current_lr:.2e}"
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
