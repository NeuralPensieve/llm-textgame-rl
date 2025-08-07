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
        # Group experiences by (env_idx, episode_id)
        grouped_experiences = defaultdict(list)
        for exp in rollout_buffer:
            grouped_experiences[(exp["env_idx"], exp["episode_id"])].append(exp)

        all_advantages = []
        all_returns = []

        for key, experiences in grouped_experiences.items():
            if not experiences:
                continue

            advantages = []
            returns = []

            last_exp = experiences[-1]
            last_value = last_exp["value"] if last_exp["truncated"] else 0.0
            last_advantage = 0.0

            # Compute rewards and advantages in reverse order
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
                returns.append(advantage + exp["value"])
                advantages.append(advantage)

                last_value = exp["value"]
                last_advantage = advantage

            # Restore chronological order
            advantages.reverse()
            returns.reverse()

            all_advantages.extend(advantages)
            all_returns.extend(returns)

        # Convert to arrays
        advantages = np.array(all_advantages, dtype=np.float32)
        returns = np.array(all_returns, dtype=np.float32)

        # Clip advantages to limit extreme values
        advantages = np.clip(advantages, -5, 5)

        # Normalize advantages
        if len(advantages) > 1 and self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

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
        # total_kl_loss = 0
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
                    current_action_logprobs, current_values = (
                        self.policy.evaluate_actions(batch_states, batch_action_lists)
                    )

                    # Check for non-finite model outputs
                    if not (
                        self._has_any_finite_tensor(current_action_logprobs)
                        and current_values.isfinite().all()
                    ):
                        self.logger.warning(
                            "Non-finite values in model outputs. Skipping batch."
                        )
                        wandb.log({"warning": "Non-finite model outputs"})
                        continue

                    # Extract logprobs for chosen actions
                    current_logprobs = torch.stack(
                        [
                            current_action_logprobs[i][action_idx]
                            for i, action_idx in enumerate(batch_action_indices)
                        ]
                    )

                    # Compute new logits for KL loss
                    # TODO: It needs to be fixed for helpful_token case, if it is to be used again
                    # action_prompts = [
                    #     f"In game state: {state}, best action is {action}"
                    #     for state, action in zip(
                    #         batch_states, [exp["action"] for exp in batch_experiences]
                    #     )
                    # ]
                    # action_inputs = self.policy.tokenize_prompts(action_prompts)
                    # action_inputs = {
                    #     k: v.to(self.device) for k, v in action_inputs.items()
                    # }
                    # new_logits, _ = self.policy(**action_inputs)

                    # # Get old logits from rollout buffer
                    # old_logits = torch.stack(
                    #     [exp["old_logits"].to(self.device) for exp in batch_experiences]
                    # )

                    # # Check for non-finite logits
                    # if not (
                    #     new_logits.isfinite().all() and old_logits.isfinite().all()
                    # ):
                    #     self.logger.warning(
                    #         "Non-finite values in logits. Skipping batch."
                    #     )
                    #     wandb.log({"warning": "Non-finite logits"})
                    #     continue

                    # # Compute KL loss
                    # kl_loss = self.policy.get_kl_loss(new_logits, old_logits)

                    # Ensure shapes match
                    current_values = current_values.view(-1)
                    batch_returns = batch_returns.view(-1)

                    # Compute losses
                    logprob_diff = torch.clamp(
                        current_logprobs - batch_old_logprobs, -3, 3
                    )
                    ratio = torch.exp(logprob_diff)
                    ratio = torch.clamp(ratio, 0.1, 10.0)
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

                    # Check for non-finite losses
                    if not (
                        policy_loss.isfinite()
                        and value_loss.isfinite()
                        # and kl_loss.isfinite()
                    ):
                        self.logger.warning("Non-finite loss values. Skipping batch.")
                        wandb.log({"warning": "Non-finite loss values"})
                        continue

                    total_loss = (
                        policy_loss
                        + self.config.value_loss_coef * value_loss
                        # + self.config.kl_loss_coef * kl_loss
                    ) / accumulation_steps

                self.scaler.scale(total_loss).backward()

                batch_count += 1

                # Apply gradients after accumulation_steps or at the final batch
                if (batch_count % accumulation_steps == 0) or (
                    end >= len(rollout_buffer)
                ):
                    self.scaler.unscale_(self.optimizer)
                    # Clip gradients with relaxed non-finite handling
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm,
                        norm_type=2.0,
                        error_if_nonfinite=False,  # Allow clipping even with non-finite norms
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    batch_count = 0

                    # Log gradient norm
                    wandb.log(
                        {
                            "gradient_norm": (
                                total_norm.item()
                                if total_norm.isfinite()
                                else float("inf")
                            )
                        }
                    )

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                # total_kl_loss += kl_loss.item()
                num_updates += 1

                # Clear tensors to free memory
                del (
                    current_logprobs,
                    current_values,
                    ratio,
                    surr1,
                    surr2,
                    policy_loss,
                    value_loss,
                    total_loss,
                    logprob_diff,
                )
                torch.cuda.empty_cache()

        # Update learning rate
        self.scheduler.step()

        # Log metrics
        if num_updates > 0:
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "iteration": iteration,
                    "policy_loss": total_policy_loss / num_updates,
                    "value_loss": total_value_loss / num_updates,
                    # "kl_loss": total_kl_loss / num_updates,
                    "learning_rate": current_lr,
                    "cumulative_episodes": self.cumulative_episodes,
                    "episodes_per_update": unique_episodes,
                }
            )

            self.logger.info(
                f"Policy Loss: {total_policy_loss/num_updates:.4f}, "
                f"Value Loss: {total_value_loss/num_updates:.4f}, "
                # f"KL Loss: {total_kl_loss/num_updates:.4f}, "
                f"LR: {current_lr:.2e}, "
                f"Cumulative Episodes: {self.cumulative_episodes}, "
                f"Episodes in Update: {unique_episodes}"
                f"Scoring: {'action_tokens' if self.config.use_action_token_scoring else 'helpful_token'}"
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
