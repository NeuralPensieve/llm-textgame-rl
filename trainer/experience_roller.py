import re
import torch
import numpy as np
from typing import Tuple, List, Dict, NamedTuple
from tqdm import tqdm
from torch.amp import autocast

from env import TextWorldEnvironment
from config import PPOConfig
from models import LLMPolicy
from helper import tokenize_actions_for_trie, generate_mask, format_prompt
from helper import TokenizerHelper, Trie


class EpisodeStats(NamedTuple):
    length: int
    reward: float
    completed: bool
    game_log: str


class ExperienceRoller:
    """
    Handles running parallel environments and collecting token-level experiences.
    This is the unified, core logic for both training rollouts and evaluation.
    """
    def __init__(self, policy: LLMPolicy, config: PPOConfig, device: torch.device, tokenizer_helper: TokenizerHelper):
        self.policy = policy
        self.config = config
        self.device = device
        self.tokenizer_helper = tokenizer_helper

    def run(self, num_episodes: int, temperature: float, is_eval_mode: bool) -> Tuple[List[Dict], List[EpisodeStats]]:
        """
        Runs N parallel rollouts until each episode is complete.
        Args:
            num_episodes (int): Number of parallel environments to run.
            temperature (float): Sampling temperature for action selection.
            is_eval_mode (bool): Whether to run in evaluation mode (zero temperature for greedy actions).
        Returns:
            Tuple containing:
                - List of dictionaries with token-level experiences.
                - List of EpisodeStats for each completed episode.
        """
        # --- OUTER LOOP SETUP ---
        envs = [TextWorldEnvironment(config=self.config, is_eval_env=is_eval_mode) for _ in range(num_episodes)]
        game_observations = [env.reset() for env in envs]
        
        rollout_buffer = []
        finished_episodes = []
        
        # Manages which environments (episodes) are completely done.
        active_envs_mask = [True] * num_episodes
        current_lengths = np.zeros(num_episodes, dtype=int)
        num_sample_games = min(num_episodes, 5) if is_eval_mode else 0
        game_logs = [[] for _ in range(num_sample_games)]
        if is_eval_mode:
            for i in range(num_sample_games):
                game_logs[i].append(f"\n=== SAMPLE GAME {i + 1} ===")
        
        # --- OUTER LOOP: Manages the entire set of episodes ---
        with tqdm(total=num_episodes, desc="Completing episodes (Token-level)") as pbar:
            while any(active_envs_mask):
                
                # --- 1. TURN SETUP (Runs once per game turn) ---
                active_indices = [i for i, active in enumerate(active_envs_mask) if active]
                
                # Get actions and create a Trie for each active environment.
                available_actions = [envs[i].get_valid_actions() for i in active_indices]
                tokenized_for_tries = [tokenize_actions_for_trie(act, self.tokenizer_helper.tokenizer) for act in available_actions]
                start_token_id = self.tokenizer_helper.tokenizer.encode(">", add_special_tokens=False)[0]

                tries = []
                for token_ids in tokenized_for_tries:
                    trie = Trie(start_token_id)
                    for action in token_ids:
                        trie.insert(action)
                    tries.append(trie)
                
                # Manages which environments have completed the CURRENT turn's action.
                turn_complete_mask = [False] * len(active_indices)
                # Stores the buffer indices for the current turn to update the final reward later.
                turn_buffer_indices = [[] for _ in range(len(active_indices))]
                partial_commands_str = [""] * len(active_indices)

                # Tokenize the initial game observations to create the starting tensors.
                initial_prompts = [format_prompt(game_observations[i]) for i in active_indices]
                padded_batch = self.tokenizer_helper.tokenize_prompts(initial_prompts)
                
                # This list will hold the token sequences for all envs active in this turn
                turn_sequences = [seq.tolist() for seq in padded_batch["input_ids"]]

                # --- INNER LOOP: Generates one multi-token action for all active envs ---
                while not all(turn_complete_mask):
                    
                    # --- 2. PREPARE TOKEN BATCH ---
                    active_turn_indices = [i for i, complete in enumerate(turn_complete_mask) if not complete]

                    # We need to re-pad the active sequences at each step
                    active_sequences = [turn_sequences[i] for i in active_turn_indices]
                    padded_batch = self.tokenizer_helper.tokenizer.pad({"input_ids": active_sequences}, padding="longest", return_tensors="pt")
                    batch_input_ids = padded_batch["input_ids"].to(self.device)
                    batch_attention_mask = padded_batch["attention_mask"].to(self.device)
                    
                    # --- 3. GET POLICY OUTPUT ---
                    batch_tries = [tries[i] for i in active_turn_indices]
                    token_mask = generate_mask(batch_tries, self.tokenizer_helper.tokenizer.vocab_size).to(self.policy.device)

                    batch_input_ids = batch_input_ids.to(self.policy.device)
                    batch_attention_mask = batch_attention_mask.to(self.policy.device)
                    
                    with torch.no_grad(), autocast(self.policy.device.type):
                        logits, values = self.policy.forward(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    
                    last_token_logits = logits[:, -1, :] + token_mask.to(dtype=logits.dtype)
                    
                    # --- 4. SAMPLE & LOG ---
                    temp_to_use = 0.0 if is_eval_mode else temperature
                    next_token_ids, chosen_log_probs = self._temperature_sampling(last_token_logits, temp_to_use)

                    for i, (token_id, logprob) in enumerate(zip(next_token_ids, chosen_log_probs)):
                        original_turn_idx = active_turn_indices[i]
                        
                        path = tries[original_turn_idx].update_head(token_id)

                        # --- Only store experiences in the buffer during training ---
                        if not is_eval_mode:
                            state_for_log = (game_observations[active_indices[original_turn_idx]], partial_commands_str[original_turn_idx])

                            # Create a log entry for EACH token in the generated path
                            for j, path_token_id in enumerate(path):
                                log_entry = {
                                    "env_idx": active_indices[original_turn_idx],
                                    "state": state_for_log,
                                    "action": path_token_id,
                                    "old_logprob": logprob if j == 0 else 0.0, # Logprob is non-zero only for the sampled token
                                    "value": values[i].item(),
                                    "reward": 0.0,
                                    "done": False,
                                }
                                rollout_buffer.append(log_entry)
                                turn_buffer_indices[original_turn_idx].append(len(rollout_buffer) - 1)

                        # Update the master list of sequences and the partial command string
                        turn_sequences[original_turn_idx].extend(path)
                        partial_commands_str[original_turn_idx] += self.tokenizer_helper.tokenizer.decode(path)
                        
                        # If that action leads to the EOS token, mark the turn as complete
                        if tries[original_turn_idx].head.value == self.tokenizer_helper.tokenizer.eos_token_id:
                            turn_complete_mask[original_turn_idx] = True

                # --- 6. EXECUTE COMPLETED ACTIONS IN ENVIRONMENT ---
                completed_actions_text = [s.replace(self.tokenizer_helper.tokenizer.eos_token, "").strip() for s in partial_commands_str]
                
                for i in range(len(active_indices)):
                    original_env_idx = active_indices[i]
                    action_text = completed_actions_text[i]
                    # Capture the state BEFORE the action is taken
                    current_obs_for_log = None
                    if original_env_idx < num_sample_games:
                        current_obs_for_log = game_observations[original_env_idx]

                    # Execute the action in the environment
                    next_obs, reward, done, info = envs[original_env_idx].step(action_text)

                    current_lengths[original_env_idx] += 1

                    # Create the new, simplified log entry
                    if original_env_idx < num_sample_games:
                        step_number = current_lengths[original_env_idx]
                        log_entry = [
                            f"--- Step {step_number} ---",
                            f"STATE:\n{current_obs_for_log}",
                            f"CHOSEN ACTION: {action_text}",
                            f"REWARD: {reward}\n",
                        ]
                        game_logs[original_env_idx].extend(log_entry)
                    
                    # --- 7. UPDATE FINAL REWARD & OUTER LOOP MASK ---
                    if not is_eval_mode and turn_buffer_indices[i]:
                        final_buffer_idx = turn_buffer_indices[i][-1]
                        rollout_buffer[final_buffer_idx]['reward'] = reward
                        # The 'done' flag for GAE should reflect truncation
                        is_truncated = current_lengths[original_env_idx] >= self.config.num_steps
                        rollout_buffer[final_buffer_idx]['done'] = done or is_truncated
                    
                    game_observations[original_env_idx] = next_obs
                    
                    if done or current_lengths[original_env_idx] >= self.config.num_steps:
                        active_envs_mask[original_env_idx] = False
                        game_won = info.get("won", False)
                        game_log_str = None
                        if original_env_idx < num_sample_games:
                            game_log_str = "\n".join(game_logs[original_env_idx])

                        # The length for stats is the number of turns taken
                        episode_turn_length = current_lengths[original_env_idx]
                        total_reward = sum(exp['reward'] for exp in rollout_buffer if exp['env_idx'] == original_env_idx)

                        finished_episodes.append(EpisodeStats(
                            length=episode_turn_length,
                            reward=total_reward,
                            completed=game_won,
                            game_log=game_log_str
                        ))
                        pbar.update(1)
        
        for env in envs: 
            env.close()

        return rollout_buffer, finished_episodes

    def _temperature_sampling(self, logits: torch.Tensor, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        """Applies temperature sampling to logits and returns token IDs and their log probabilities."""
        if temperature < 1e-5:
            next_token_ids = torch.argmax(logits, dim=-1)
        else:
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = torch.gather(log_probs, 1, next_token_ids.view(-1, 1)).squeeze(1)
        
        return next_token_ids.cpu().numpy(), chosen_log_probs.cpu().numpy()