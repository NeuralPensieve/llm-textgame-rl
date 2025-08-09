import torch
import numpy as np
import wandb
from typing import Tuple, List, Dict
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

    def _run_single_episode(self, episode_idx: int, log_details: bool = False) -> Dict:
        """Run a single episode and return results with optional detailed logging"""
        env = TextWorldEnvironment(difficulty=self.config.difficulty)
        state = env.reset()
        
        game_log = []
        if log_details:
            game_log.append(f"=== SAMPLE GAME {episode_idx + 1} ===")
            game_log.append(f"Initial State: {state}")
            game_log.append("")

        step_count = 0
        total_reward = 0.0
        max_steps = self.config.num_steps
        done = False

        while step_count < max_steps:
            # Get valid actions
            actions = env.get_valid_actions()

            # Get policy evaluation
            with torch.no_grad(), autocast(self.device.type):
                action_logprobs, value = self.policy.evaluate_for_rollout(
                    [state], [actions]
                )

            # Select best action deterministically (argmax)
            action_idx = np.argmax(action_logprobs[0])
            chosen_action = actions[action_idx]

            if log_details:
                # Log the step details
                game_log.append(f"Step {step_count + 1}:")
                game_log.append(f"  Available Actions: {actions}")
                game_log.append(
                    f"  Action Probabilities: {[f'{np.exp(prob):.3e}' for prob in action_logprobs[0]]}"
                )
                game_log.append(f"  Chosen Action: {chosen_action}")
                game_log.append(f"  State Value: {value[0]:.3f}")

            # Take the action
            next_state, reward, done, info = env.step(chosen_action)
            total_reward += reward
            step_count += 1

            if log_details:
                game_log.append(f"  Reward: {reward}")
                game_log.append(f"  Done: {done}")
                game_log.append(f"  Next State: {next_state}")
                game_log.append("")

            if done:
                if log_details:
                    game_log.append(f"Game completed in {step_count} steps!")
                break
            else:
                # Update state with action context for next iteration
                state = state + f"\n\naction taken: {chosen_action}\n\n" + next_state

        # Add game summary if logging details
        if log_details:
            game_log.append(f"=== GAME {episode_idx + 1} SUMMARY ===")
            game_log.append(f"Total Steps: {step_count}")
            game_log.append(f"Total Reward: {total_reward:.3f}")
            game_log.append(f"Completed: {'Yes' if done else 'No (truncated)'}")
            game_log.append("=" * 50)
            game_log.append("")

        # Clean up
        del env

        return {
            "length": step_count,
            "reward": total_reward,
            "completed": done,
            "game_log": "\n".join(game_log) if log_details else None
        }

    def run_evaluation(self, iteration: int) -> Tuple[float, float]:
        """Run evaluation rollouts with deterministic policy (no epsilon-greedy)"""
        self.logger.info("Running evaluation...")
        
        # Run episodes for both evaluation metrics and sample game logging
        num_eval_episodes = 20  # Episodes for metrics
        num_sample_games = 5    # Episodes for detailed logging
        
        # First run sample games (with detailed logging)
        wandb.log({"sample_games_log": f"Generating {num_sample_games} sample games..."})
        
        sample_games = []
        all_episodes = []  # Track ALL episodes (completed or not)
        
        for i in tqdm(range(num_sample_games), desc="Generating sample games"):
            episode_result = self._run_single_episode(i, log_details=True)
            sample_games.append(episode_result["game_log"])
            all_episodes.append(episode_result)
        
        # Then run additional episodes for evaluation metrics (without detailed logging)
        remaining_episodes = num_eval_episodes - num_sample_games
        
        for i in tqdm(range(remaining_episodes), desc="Running additional evaluation episodes"):
            episode_result = self._run_single_episode(num_sample_games + i, log_details=False)
            all_episodes.append(episode_result)
        
        # Calculate evaluation metrics from ALL episodes (completed or not)
        if all_episodes:
            avg_episode_length = np.mean([ep["length"] for ep in all_episodes])
            avg_episode_reward = np.mean([ep["reward"] for ep in all_episodes])
            
            # Count how many episodes completed successfully
            completed_episodes = [ep for ep in all_episodes if ep["completed"]]
            sample_completed = sum(1 for ep in all_episodes[:num_sample_games] if ep["completed"])
            
            self.logger.info(
                f"Evaluation completed: {len(all_episodes)} total episodes, "
                f"{len(completed_episodes)} naturally completed, "
                f"Sample games completed: {sample_completed}/{num_sample_games}, "
                f"Avg Length: {avg_episode_length:.2f}, "
                f"Avg Reward: {avg_episode_reward:.4f}"
            )
        else:
            avg_episode_length = 0.0
            avg_episode_reward = 0.0
            self.logger.warning("ALERT: No episodes run during evaluation!")

        # Log sample games to file
        all_games_text = "\n\n".join(sample_games)
        with open(f"evaluations/{wandb.run.name}.txt", "a") as f:
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Evaluation Metrics - Total Episodes: {len(all_episodes)}, ")
            completed_count = sum(1 for ep in all_episodes if ep["completed"])
            f.write(f"Naturally Completed: {completed_count}, ")
            f.write(f"Sample Games Completed: {sum(1 for ep in all_episodes[:num_sample_games] if ep['completed'])}/{num_sample_games}, ")
            f.write(f"Avg Length: {avg_episode_length:.2f}, Avg Reward: {avg_episode_reward:.4f}\n\n")
            f.write(all_games_text)
            f.write("\n\n")

        torch.cuda.empty_cache()

        return avg_episode_length, avg_episode_reward