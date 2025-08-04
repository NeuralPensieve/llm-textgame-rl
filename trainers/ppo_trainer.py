import torch
import torch.nn.functional as F
import numpy as np
import random
import datetime
from typing import List, Dict, Tuple
import logging
import os
from tqdm import tqdm
import wandb
from torch.amp import GradScaler, autocast

from env import TextWorldEnvironment
from config import PPOConfig
from models import LLMPolicy


class PPOTextWorldTrainer:
    """Simplified PPO trainer for TextWorld"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup logging and wandb
        self._setup_logging()

        # Clear games folder
        if not os.path.exists("./games"):
            os.makedirs("./games")
        else:
            if not self.config.reuse_seed:
                for f in os.listdir("./games"):
                    os.remove(os.path.join("./games", f))

        # Create environments and policy
        self.envs = [
            TextWorldEnvironment(
                seed=self.config.env_seed, reuse_seed=self.config.reuse_seed
            )
            for _ in range(config.num_envs)
        ]
        self.policy = LLMPolicy(config).to(self.device)

        # Disable cache for transformer models during training
        if hasattr(self.policy, "config"):
            self.policy.config.use_cache = False

        # Improved optimizer with separate learning rates
        param_groups = self.policy.get_separate_parameter_groups()
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_iterations, eta_min=1e-6
        )

        # Initialize GradScaler for FP16 training
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # Training state
        self.iteration = 0
        self.epsilon = config.epsilon

        os.makedirs("checkpoints", exist_ok=True)

    def _setup_logging(self):
        """Setup logging and wandb"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        run_name = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        wandb.init(project="textworld-llm-ppo", name=run_name, config=vars(self.config))

    def run_evaluation(self) -> Tuple[float, float]:
        """Run evaluation rollouts with deterministic policy (no epsilon-greedy)"""
        self.logger.info("Running evaluation...")

        # Create fresh environments for evaluation
        eval_envs = [TextWorldEnvironment() for _ in range(self.config.num_envs)]
        eval_states = [env.reset() for env in eval_envs]

        episode_lengths = [0] * len(eval_envs)
        episode_rewards = [0.0] * len(eval_envs)
        completed_episodes = []
        episode_ids = [0] * len(eval_envs)

        # Run for the same number of steps as training rollouts
        for step in tqdm(range(self.config.num_steps), desc="Evaluating"):
            # Get valid actions for all environments
            batch_states = []
            batch_actions = []

            for i, (env, state) in enumerate(zip(eval_envs, eval_states)):
                actions = env.get_valid_actions()
                batch_states.append(state)
                batch_actions.append(actions)

            # Evaluate actions deterministically (no epsilon-greedy)
            with torch.no_grad(), autocast(self.device.type):
                action_logprobs, _ = self.policy.evaluate_for_rollout(
                    batch_states, batch_actions
                )

            # Step each environment with deterministic action selection
            new_states = []
            for i, (env, state, actions, logprobs) in enumerate(
                zip(eval_envs, eval_states, batch_actions, action_logprobs)
            ):
                # Select best action deterministically (argmax)
                action_idx = np.argmax(logprobs)
                chosen_action = actions[action_idx]

                next_state, reward, done, info = env.step(chosen_action)

                # Update episode tracking
                episode_lengths[i] += 1
                episode_rewards[i] += reward

                if done:
                    # Store completed episode
                    completed_episodes.append(
                        {"length": episode_lengths[i], "reward": episode_rewards[i]}
                    )

                    # Reset episode tracking
                    episode_lengths[i] = 0
                    episode_rewards[i] = 0.0
                    episode_ids[i] += 1

                    # Reset environment
                    new_states.append(env.reset())
                else:
                    # Continue episode with action context
                    updated_state = (
                        state
                        + "\n\n"
                        + f"action taken: {chosen_action}\n\n"
                        + next_state
                    )
                    new_states.append(updated_state)

            eval_states = new_states

        # Include truncated episodes (episodes that didn't finish within num_steps)
        for i, (length, reward) in enumerate(zip(episode_lengths, episode_rewards)):
            if length > 0:
                completed_episodes.append({"length": length, "reward": reward})

        # Calculate evaluation metrics
        if completed_episodes:
            avg_episode_length = np.mean([ep["length"] for ep in completed_episodes])
            avg_episode_reward = np.mean([ep["reward"] for ep in completed_episodes])
        else:
            avg_episode_length = 0.0
            avg_episode_reward = 0.0

        self.logger.info(
            f"Evaluation completed: {len(completed_episodes)} episodes, "
            f"Avg Length: {avg_episode_length:.2f}, "
            f"Avg Reward: {avg_episode_reward:.4f}"
        )

        # Clean up evaluation environments
        del eval_envs
        torch.cuda.empty_cache()

        return avg_episode_length, avg_episode_reward

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

    def compute_advantages(
        self, rollout_buffer: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns with episode-level diagnostics."""
        from collections import defaultdict

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

    def ppo_update(self, rollout_buffer: List[Dict]):
        """PPO policy update"""
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
        total_kl_loss = 0
        num_updates = 0

        accumulation_steps = self.config.accumulation_steps
        effective_batch_size = min(self.config.batch_size, len(rollout_buffer))

        # PPO epochs
        ppo_step = 0
        for epoch in range(self.config.ppo_epochs):
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
                        self.has_any_infinite_tensor(current_action_logprobs)
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
                    action_prompts = [
                        f"In game state: {state}, best action is {action}"
                        for state, action in zip(
                            batch_states, [exp["action"] for exp in batch_experiences]
                        )
                    ]
                    action_inputs = self.policy.tokenize_prompts(action_prompts)
                    action_inputs = {
                        k: v.to(self.device) for k, v in action_inputs.items()
                    }
                    new_logits, _ = self.policy(**action_inputs)

                    # Get old logits from rollout buffer
                    old_logits = torch.stack(
                        [exp["old_logits"].to(self.device) for exp in batch_experiences]
                    )

                    # Check for non-finite logits
                    if not (
                        new_logits.isfinite().all() and old_logits.isfinite().all()
                    ):
                        self.logger.warning(
                            "Non-finite values in logits. Skipping batch."
                        )
                        wandb.log({"warning": "Non-finite logits"})
                        continue

                    # Compute KL loss
                    kl_loss = self.policy.get_kl_loss(new_logits, old_logits)

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
                        and kl_loss.isfinite()
                    ):
                        self.logger.warning("Non-finite loss values. Skipping batch.")
                        wandb.log({"warning": "Non-finite loss values"})
                        continue

                    # Log diagnostics
                    # wandb.log({
                    #     "ppo_step": ppo_step,
                    #     "logprob_diff_mean": logprob_diff.mean().item(),
                    #     "logprob_diff_max": logprob_diff.max().item(),
                    #     "ratio_mean": ratio.mean().item(),
                    #     "ratio_max": ratio.max().item(),
                    #     "surr1_mean": surr1.mean().item(),
                    #     "surr1_min": surr1.min().item(),
                    #     "surr2_mean": surr2.mean().item(),
                    #     "surr2_min": surr2.min().item(),
                    #     "advantages_mean": batch_advantages.mean().item(),
                    #     "advantages_max": batch_advantages.max().item(),
                    # })

                    total_loss = (
                        policy_loss
                        + self.config.value_loss_coef * value_loss
                        + self.config.kl_loss_coef * kl_loss
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
                total_kl_loss += kl_loss.item()
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

        # Decay epsilon
        self.epsilon = max(
            self.config.min_epsilon, self.epsilon * self.config.epsilon_decay
        )

        # Log metrics
        if num_updates > 0:
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "iteration": self.iteration,
                    "policy_loss": total_policy_loss / num_updates,
                    "value_loss": total_value_loss / num_updates,
                    "kl_loss": total_kl_loss / num_updates,
                    "epsilon": self.epsilon,
                    "learning_rate": current_lr,
                }
            )

            self.logger.info(
                f"Policy Loss: {total_policy_loss/num_updates:.4f}, "
                f"Value Loss: {total_value_loss/num_updates:.4f}, "
                f"KL Loss: {total_kl_loss/num_updates:.4f}, "
                f"LR: {current_lr:.2e}, "
                f"Epsilon: {self.epsilon:.4f}, "
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self):
        """Main training loop"""
        self.logger.info("Starting PPO training...")

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration

            # Run evaluation
            if iteration % self.config.eval_interval == 0:
                eval_length, eval_reward = self.run_evaluation()
                wandb.log(
                    {
                        "iteration": iteration,
                        "eval_avg_episode_length": eval_length,
                        "eval_avg_episode_reward": eval_reward,
                    }
                )

            # Update to receive additional metrics
            rollout_buffer, episode_lengths, episode_rewards = self.collect_rollouts()

            # Calculate metrics
            rewards = [exp["reward"] for exp in rollout_buffer]
            avg_reward = np.mean(rewards) if rewards else 0.0
            avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
            avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            total_episode_reward = np.sum(episode_rewards) if episode_rewards else 0.0

            # Update policy
            self.ppo_update(rollout_buffer)

            # Log iteration metrics
            wandb.log(
                {
                    "iteration": iteration,
                    "avg_reward": avg_reward,
                    "avg_episode_length": avg_episode_length,
                    "avg_episode_reward": avg_episode_reward,
                    "total_episode_reward": total_episode_reward,
                    "total_experiences": len(rollout_buffer),
                }
            )

            if iteration % self.config.log_interval == 0:
                self.logger.info(
                    f"Iteration {iteration}: Avg Reward: {avg_reward:.4f}, "
                    f"Avg Episode Length: {avg_episode_length:.2f}, "
                    f"Avg Episode Reward: {avg_episode_reward:.4f}, "
                    f"Total Episode Reward: {total_episode_reward:.4f}, "
                )

            # Save checkpoint
            if (iteration + 1) % self.config.save_interval == 0 and iteration > 10:
                self.save_checkpoint(iteration)

        wandb.finish()
        self.logger.info("Training completed!")

    def save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),  # Save scaler state
            "config": self.config,
            "epsilon": self.epsilon,
        }

        run_name = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

        checkpoint_path = f"checkpoints/{run_name}_iter_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(
            checkpoint.get("scaler_state_dict", {})
        )  # Load scaler state
        self.iteration = checkpoint["iteration"]
        self.epsilon = checkpoint["epsilon"]
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def has_any_infinite_tensor(self, nested_tensor_list):
        for sublist in nested_tensor_list:
            for tensor in sublist:
                if torch.isfinite(tensor).all():
                    return True
        return False
