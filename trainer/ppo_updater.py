import torch
import torch.nn.functional as F
import numpy as np
import wandb
import gc
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.amp import autocast
from tqdm import tqdm

from config import PPOConfig, DynamicConfigManager
from models import LLMPolicy
from helper import TokenizerHelper


class PPOUpdater:
    """
    Handles PPO policy updates for a token-level generative policy.
    """
    def __init__(self, config: PPOConfig, policy: LLMPolicy, optimizer, scheduler, scaler, device, logger, tokenizer_helper: TokenizerHelper):
        self.config = config
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.logger = logger
        self.tokenizer_helper = tokenizer_helper # Store the helper
        self.dynamic_config = DynamicConfigManager(config, logger)
        self.kl_coef = config.kl_coef if config.use_kl_penalty else 0.0

    def compute_advantages(self, rollout_buffer: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Generalized Advantage Estimation (GAE) over the token-level buffer.
        """
        experiences_by_env = defaultdict(list)
        for exp in rollout_buffer:
            experiences_by_env[exp["env_idx"]].append(exp)

        advantages_flat = [0] * len(rollout_buffer)
        returns_flat = [0] * len(rollout_buffer)

        for env_idx in sorted(experiences_by_env.keys()):
            experiences = experiences_by_env[env_idx]
            
            # The value after the last state is 0 if the episode is done.
            next_value = 0.0

            # Bootstrap if the episode was truncated AND it did not simultaneously finish.
            if experiences[-1]["truncated"] and not experiences[-1]["finished"]:
                next_value = experiences[-1]["value"]

            last_advantage = 0.0

            # Find original indices to prevent errors with non-unique experiences
            original_indices = [i for i, exp in enumerate(rollout_buffer) if exp["env_idx"] == env_idx]
            
            for i, exp in reversed(list(enumerate(experiences))):
                delta = exp["reward"] + self.config.gamma * next_value - exp["value"]
                advantage = delta + self.config.gamma * self.config.gae_lambda * last_advantage
                
                exp_index = original_indices[i]
                returns_flat[exp_index] = advantage + exp["value"]
                advantages_flat[exp_index] = advantage

                next_value = exp["value"]
                last_advantage = advantage

        advantages = np.array(advantages_flat, dtype=np.float32)
        returns = np.array(returns_flat, dtype=np.float32)

        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update(self, rollout_buffer: List[Dict], iteration: int):
        """
        Performs the PPO update on a batch of token-level experiences.
        """
        self.logger.info(f"\n--- Starting PPO Update for Iteration {iteration} ---")
        
        advantages, returns = self.compute_advantages(rollout_buffer)

        # # =================== ADD THIS BLOCK ===================
        # # Normalize the returns to make the value function's target more stable
        # returns_mean = returns.mean()
        # returns_std = returns.std() + 1e-8 # Add epsilon for numerical stability
        # returns = (returns - returns_mean) / returns_std
        # # ======================================================

        # ===== VALUE FUNCTION DIAGNOSTICS =====
        if iteration % 5 == 0:  # Every 5 iterations
            with torch.no_grad():
                self.value_diagnostics(rollout_buffer, advantages, returns, iteration)

        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_logprobs = torch.FloatTensor([exp["old_logprob"] for exp in rollout_buffer]).to(self.device)

        # --- Initialize logging variables ---
        total_policy_loss, total_value_loss, total_combined_loss = 0, 0, 0
        num_updates = 0
        epoch_kl_values = []

        # PPO epochs
        for epoch in tqdm(range(self.config.ppo_epochs), desc="Running PPO Epochs"):
            indices = torch.randperm(len(rollout_buffer))

            for start in range(0, len(rollout_buffer), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                # Get batch data from the token-level buffer
                batch_experiences = [rollout_buffer[i] for i in batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]

                # Prepare inputs for the new policy evaluation method
                batch_composite_states = [exp["state"] for exp in batch_experiences]
                batch_chosen_tokens = [exp["action"] for exp in batch_experiences]

                # Only the get the first sampled token, not the following deterministic tokens
                batch_first_tokens = [tokens[0] for tokens in batch_chosen_tokens]

                # Forward pass with the new method
                with autocast(self.device.type):
                    current_logprobs, current_values = self.policy.evaluate_tokens(
                        batch_composite_states, batch_first_tokens
                    )

                    # --- KL Divergence Calculation ---
                    kl_div = torch.tensor(0.0, device=self.device)
                    if self.config.use_kl_penalty:
                        # Sanity check on the first batch of the first iteration
                        if iteration == 0 and epoch == 0 and start == 0:
                            self._kl_sanity_check(batch_composite_states, batch_first_tokens)

                        # Get logprobs from the reference model
                        reference_logprobs = self.policy.get_reference_token_logprobs(
                            batch_composite_states, batch_first_tokens
                        )
                        kl_div = self.policy.compute_kl_divergence(current_logprobs, reference_logprobs)
                        epoch_kl_values.append(kl_div.item())

                    # --- PPO Loss Calculation ---
                    current_values = current_values.view(-1)
                    ratio = torch.exp(current_logprobs - batch_old_logprobs)

                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.epsilon_clip, 1 + self.config.epsilon_clip) * batch_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(current_values.view(-1), batch_returns)


                    total_loss = (
                        policy_loss
                        + self.dynamic_config.get('value_loss_coef') * value_loss
                        + self.kl_coef * kl_div
                    ) / self.config.accumulation_steps

                self.scaler.scale(total_loss).backward()

                # Gradient accumulation step
                if ((start // self.config.batch_size) + 1) % self.config.accumulation_steps == 0 or end >= len(rollout_buffer):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                # --- Update logging trackers ---
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_combined_loss += total_loss.item() * self.config.accumulation_steps
                num_updates += 1

        self.scheduler.step()

        if num_updates > 0:
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({
                "iteration": iteration,
                "policy_loss": total_policy_loss / num_updates,
                "value_loss": total_value_loss / num_updates,
                "total_loss": total_combined_loss / num_updates,
                "learning_rate": current_lr,
                "kl_divergence": np.mean(epoch_kl_values) if epoch_kl_values else 0.0,
                "dynamic/value_loss_coef": self.dynamic_config.get('value_loss_coef'),
            })
            self.logger.info(
                f"Update Complete. Policy Loss: {total_policy_loss/num_updates:.4f}, "
                f"Value Loss: {total_value_loss/num_updates:.4f}"
            )
        self.logger.info("--- PPO Update Finished ---")

        del advantages
        del returns
        del old_logprobs
    
        # Run garbage collection and clear cache
        gc.collect()
        torch.cuda.empty_cache()

    def _kl_sanity_check(self, states, tokens):
        """
        On the first batch, verifies that the KL divergence between the policy
        and its reference copy in eval mode is zero.
        """
        self.logger.info("Performing initial KL divergence sanity check...")
        self.policy.eval()
        with torch.no_grad():
            eval_logprobs, _ = self.policy.evaluate_tokens(states, tokens)
            ref_logprobs = self.policy.get_reference_token_logprobs(states, tokens)
            initial_kl = self.policy.compute_kl_divergence(eval_logprobs, ref_logprobs)
        
        # Allow for a tiny tolerance for floating point artifacts
        assert torch.allclose(initial_kl, torch.tensor(0.0), atol=1e-6), \
            f"Initial KL divergence in eval mode is not zero: {initial_kl.item()}"
        self.logger.info("✅ Initial KL divergence sanity check passed.")
        self.policy.train() # Restore train mode

    def value_diagnostics(self, rollout_buffer: List[Dict], advantages: np.ndarray, returns: np.ndarray, iteration: int, print_output: bool = False):
        # Get values from buffer
        values = np.array([exp["value"] for exp in rollout_buffer])
        rewards = np.array([exp["reward"] for exp in rollout_buffer])
        
        # 1. REWARD DIAGNOSTICS (using the rewards variable now!)
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        reward_min = rewards.min()
        reward_max = rewards.max()
        non_zero_rewards = rewards[rewards != 0]
        pct_non_zero = len(non_zero_rewards) / len(rewards) * 100 if len(rewards) > 0 else 0

        if print_output:
            print(f"\n===== REWARD DIAGNOSTICS (Iter {iteration}) =====")
            print(f"Immediate Rewards: mean={reward_mean:.4f}, std={reward_std:.4f}")
            print(f"Reward range: [{reward_min:.4f}, {reward_max:.4f}]")
            print(f"Non-zero rewards: {pct_non_zero:.1f}% ({len(non_zero_rewards)}/{len(rewards)})")
            if len(non_zero_rewards) > 0:
                print(f"Non-zero reward mean: {non_zero_rewards.mean():.4f}")
        
        # 2. Value function accuracy metrics
        value_mean = values.mean()
        value_std = values.std()
        returns_mean = returns.mean()
        returns_std = returns.std()
        
        # 3. Correlation between values and returns
        if len(values) > 1:
            value_return_corr = np.corrcoef(values, returns)[0, 1]
            # Also check correlation between values and immediate rewards
            value_reward_corr = np.corrcoef(values, rewards)[0, 1]
        else:
            value_return_corr = 0.0
            value_reward_corr = 0.0
        
        # 4. Explained variance (key metric!)
        ev = 1 - (returns - values).var() / (returns.var() + 1e-8)
        
        # 5. Value function bias
        bias = (values - returns).mean()
        
        # 6. Check if values are collapsing or exploding
        values_range = values.max() - values.min()
        
        if print_output:
            print(f"\n===== VALUE FUNCTION DIAGNOSTICS =====")
            print(f"Values: mean={value_mean:.4f}, std={value_std:.4f}, range={values_range:.4f}")
            print(f"Returns: mean={returns_mean:.4f}, std={returns_std:.4f}")
            print(f"Correlation(Values, Returns): {value_return_corr:.4f}")
            print(f"Correlation(Values, Immediate Rewards): {value_reward_corr:.4f}")
            print(f"Explained Variance: {ev:.4f} (negative = harmful, 1.0 = perfect)")
            print(f"Value Bias: {bias:.4f}")
        
        # 7. Check advantages distribution
        adv_mean = advantages.mean()
        adv_std = advantages.std() 
        if print_output:
            print(f"Advantages: mean={adv_mean:.4f}, std={adv_std:.4f}")
        
        # 8. Episode-ending analysis
        terminal_rewards = []
        terminal_values = []
        for i, exp in enumerate(rollout_buffer):
            if exp["finished"] or exp["truncated"]:
                terminal_rewards.append(rewards[i])
                terminal_values.append(values[i])
        
        if terminal_rewards and print_output:
            print(f"\n===== TERMINAL STATE ANALYSIS =====")
            print(f"Terminal rewards: mean={np.mean(terminal_rewards):.4f}")
            print(f"Terminal values predicted: mean={np.mean(terminal_values):.4f}")
            print(f"Value prediction error at terminal: {np.mean(terminal_values) - np.mean(terminal_rewards):.4f}")
        
        # Log to wandb
        wandb.log({
            "iteration": iteration,
            "value_diagnostics/explained_variance": ev,
            "value_diagnostics/value_return_correlation": value_return_corr,
            "value_diagnostics/value_reward_correlation": value_reward_corr,
            "value_diagnostics/bias": bias,
            "value_diagnostics/value_mean": value_mean,
            "value_diagnostics/value_std": value_std,
            "value_diagnostics/returns_mean": returns_mean,
            "value_diagnostics/returns_std": returns_std,
            "value_diagnostics/value_range": values_range,
            "reward_diagnostics/reward_mean": reward_mean,
            "reward_diagnostics/reward_std": reward_std,
            "reward_diagnostics/pct_non_zero": pct_non_zero,
        })
