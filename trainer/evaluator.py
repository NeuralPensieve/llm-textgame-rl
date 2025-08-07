import torch
import numpy as np
import wandb
from typing import Tuple, List
from tqdm import tqdm
from torch.amp import autocast

from env import TextWorldEnvironment


class Evaluator:
    """Handles policy evaluation and sample game generation"""

    def __init__(self, policy, config, device, logger):
        self.policy = policy
        self.config = config
        self.device = device
        self.logger = logger

    def _generate_sample_games(self, iteration: int, num_samples: int = 5) -> List[str]:
        """Generate sample games for logging to wandb"""
        wandb.log({"sample_games_log": f"Generating {num_samples} sample games..."})

        sample_games = []

        for game_idx in range(num_samples):
            # Create a fresh environment for this sample game
            sample_env = TextWorldEnvironment(difficulty=self.config.difficulty)
            state = sample_env.reset()

            game_log = []
            game_log.append(f"=== SAMPLE GAME {game_idx + 1} ===")
            game_log.append(f"Initial State: {state}")
            game_log.append("")

            step_count = 0
            total_reward = 0.0
            max_steps = self.config.num_steps  # Use same limit as regular evaluation

            while step_count < max_steps:
                # Get valid actions
                actions = sample_env.get_valid_actions()

                # Get policy evaluation
                with torch.no_grad(), autocast(self.device.type):
                    action_logprobs, value = self.policy.evaluate_for_rollout(
                        [state], [actions]
                    )

                # Select best action deterministically (argmax)
                action_idx = np.argmax(action_logprobs[0])
                chosen_action = actions[action_idx]

                # Log the step
                game_log.append(f"Step {step_count + 1}:")
                game_log.append(f"  Available Actions: {actions}")
                game_log.append(
                    f"  Action Probabilities: {[f'{np.exp(prob):.3e}' for prob in action_logprobs[0]]}"
                )
                game_log.append(f"  Chosen Action: {chosen_action}")
                game_log.append(f"  State Value: {value[0]:.3f}")

                # Take the action
                next_state, reward, done, info = sample_env.step(chosen_action)
                total_reward += reward
                step_count += 1

                game_log.append(f"  Reward: {reward}")
                game_log.append(f"  Done: {done}")
                game_log.append(f"  Next State: {next_state}")
                game_log.append("")

                if done:
                    game_log.append(f"Game completed in {step_count} steps!")
                    break
                else:
                    # Update state with action context for next iteration
                    state = (
                        state + f"\n\naction taken: {chosen_action}\n\n" + next_state
                    )

            # Add game summary
            game_log.append(f"=== GAME {game_idx + 1} SUMMARY ===")
            game_log.append(f"Total Steps: {step_count}")
            game_log.append(f"Total Reward: {total_reward:.3f}")
            game_log.append(f"Completed: {'Yes' if done else 'No (truncated)'}")
            game_log.append("=" * 50)
            game_log.append("")

            # Join all log entries for this game
            sample_games.append("\n".join(game_log))

            # Clean up
            del sample_env

        return sample_games

    def run_evaluation(self, iteration: int) -> Tuple[float, float]:
        """Run evaluation rollouts with deterministic policy (no epsilon-greedy)"""
        self.logger.info("Running evaluation...")

        # Create fresh environments for evaluation
        eval_envs = [TextWorldEnvironment(difficulty=self.config.difficulty) for _ in range(self.config.num_envs)]
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

                # if done or truncated
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

        # Generate and log sample games
        sample_games = self._generate_sample_games(iteration, num_samples=5)

        # Log all games as a plain string to wandb logs instead of as a media artifact
        all_games_text = "\n\n".join(sample_games)
        with open(f"evaluations/{wandb.run.name}.txt", "a") as f:
            f.write(f"Iteration: {iteration}\n\n")
            f.write(all_games_text)

        # Clean up evaluation environments
        del eval_envs
        torch.cuda.empty_cache()

        return avg_episode_length, avg_episode_reward
