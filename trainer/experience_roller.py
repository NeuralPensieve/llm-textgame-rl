import re
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, NamedTuple, Optional
from tqdm import tqdm
from torch.amp import autocast

from env import TextWorldEnvironment
from config import PPOConfig
from models import LLMPolicy
from helper import tokenize_actions_for_trie, generate_mask, format_prompt
from helper import TokenizerHelper, Trie


class EpisodeStats(NamedTuple):
    """Stores statistics for a single completed episode."""
    length: int
    reward: float
    completed: bool
    game_log: str
    action_log_probs: List[float]


@dataclass
class GenerationResult:
    """Stores the rich output of the ActionGenerator for a single turn."""
    completed_actions: List[str]
    step_data: List[List[Tuple[List[int], float, float]]]
    turn_log_probs: List[float]
    valid_per_step: List[List[List[int]]]


class ParallelEnvManager:
    """Manages a batch of parallel TextWorld environments and their states."""
    def __init__(self, config: PPOConfig, is_eval_env: bool, existing_envs: Optional[List[TextWorldEnvironment]] = None, num_episodes: Optional[int] = None):
        """
        Initializes the manager with a set of parallel environments.

        Args:
            config: Configuration object for the environment.
            is_eval_env: Flag indicating if the environments are for evaluation.
            existing_envs: An optional list of pre-initialized environments to use.
            num_episodes: The number of environments to create if existing_envs is not provided.
        """
        self.config = config
        self.is_eval_env = is_eval_env

        if existing_envs:
            self.envs = existing_envs
            self.num_episodes = len(self.envs)
        elif num_episodes:
            self.num_episodes = num_episodes
            self.envs = [
                TextWorldEnvironment(config=self.config, is_eval_env=is_eval_env, game_id=i)
                for i in range(num_episodes)
            ]
        else:
            raise ValueError("Must provide either 'existing_envs' or 'num_episodes'.")

        self.observations = [env.reset() for env in self.envs]
        self.active_mask = [True] * self.num_episodes
        self.current_lengths = np.zeros(self.num_episodes, dtype=int)
        self.current_rewards = np.zeros(self.num_episodes, dtype=float)
        self.episode_action_log_probs = [[] for _ in range(self.num_episodes)]

        self.num_sample_games = min(self.num_episodes, 5) if self.is_eval_env else 0
        self.game_logs = self._initialize_game_logs()

    def _initialize_game_logs(self) -> List[List[str]]:
        logs = [[] for _ in range(self.num_sample_games)]
        if self.is_eval_env:
            for i in range(self.num_sample_games):
                logs[i].append(f"\n=== SAMPLE GAME {i + 1} ===")
        return logs

    def get_active_indices(self) -> List[int]:
        return [i for i, active in enumerate(self.active_mask) if active]

    def get_active_observations(self) -> List[str]:
        return [self.observations[i] for i in self.get_active_indices()]

    def get_valid_actions(self) -> List[List[str]]:
        return [self.envs[i].get_valid_actions() for i in self.get_active_indices()]

    def step(self, actions: List[str], turn_log_probs: List[float]) -> Tuple[List[EpisodeStats], Dict[int, float], Dict[int, bool], Dict[int, bool]]:
        completed_episodes = []
        rewards_this_turn = {}
        dones_this_turn = {}
        truncateds_this_turn = {}
        active_indices = self.get_active_indices()
        for i, action_text in enumerate(actions):
            original_env_idx = active_indices[i]
            env = self.envs[original_env_idx]
            current_obs_for_log = self.observations[original_env_idx] if original_env_idx < self.num_sample_games else None
            next_obs, reward, done, info = env.step(action_text)
            self.current_lengths[original_env_idx] += 1
            self.current_rewards[original_env_idx] += reward
            self.episode_action_log_probs[original_env_idx].append(turn_log_probs[i])
            rewards_this_turn[original_env_idx] = reward
            dones_this_turn[original_env_idx] = done
            is_max_steps = self.current_lengths[original_env_idx] >= self.config.num_steps
            truncateds_this_turn[original_env_idx] = is_max_steps and not done
            if original_env_idx < self.num_sample_games:
                step_number = self.current_lengths[original_env_idx]
                log_entry = [
                    f"--- Step {step_number} ---",
                    f"\n{current_obs_for_log}",
                    f"CHOSEN ACTION: {action_text} (Prob: {np.exp(turn_log_probs[i]):.8f})",
                    f"REWARD: {reward}\n",
                ]
                self.game_logs[original_env_idx].extend(log_entry)
            self.observations[original_env_idx] = next_obs
            if done or is_max_steps:
                self.active_mask[original_env_idx] = False
                game_won = info.get("won", False)
                game_log_str = "\n".join(self.game_logs[original_env_idx]) if original_env_idx < self.num_sample_games else None
                stats = EpisodeStats(
                    length=self.current_lengths[original_env_idx],
                    reward=self.current_rewards[original_env_idx],
                    completed=game_won,
                    game_log=game_log_str,
                    action_log_probs=self.episode_action_log_probs[original_env_idx]
                )
                completed_episodes.append(stats)
        return completed_episodes, rewards_this_turn, dones_this_turn, truncateds_this_turn

    def is_done(self) -> bool:
        return not any(self.active_mask)
        
    def close(self):
        for env in self.envs:
            env.close()


class ActionGenerator:
    """Handles token-by-token generation using the policy model."""
    def __init__(self, policy: LLMPolicy, tokenizer_helper: TokenizerHelper, device: torch.device):
        self.policy = policy
        self.tokenizer_helper = tokenizer_helper
        self.device = device
        self.eos_token_id = self.tokenizer_helper.tokenizer.eos_token_id

    def generate(self, initial_prompts: List[str], tries: List[Trie], temperature: float) -> GenerationResult:
        num_active_envs = len(initial_prompts)
        turn_complete_mask = [False] * num_active_envs
        turn_sequences = self.tokenizer_helper.tokenize_prompts(initial_prompts)["input_ids"].tolist()
        
        # This will store List[List[Tuple(path, log_prob, value)]]
        step_data = [[] for _ in range(num_active_envs)]
        valid_per_step = [[] for _ in range(num_active_envs)]

        while not all(turn_complete_mask):
            active_turn_indices = [i for i, complete in enumerate(turn_complete_mask) if not complete]
            active_sequences = [turn_sequences[i] for i in active_turn_indices]
            
            padded_batch = self.tokenizer_helper.tokenizer.pad({"input_ids": active_sequences}, return_tensors="pt")
            batch_input_ids = padded_batch["input_ids"].to(self.device)
            batch_attention_mask = padded_batch["attention_mask"].to(self.device)

            batch_tries = [tries[i] for i in active_turn_indices]
            token_mask, all_valid_tokens_batch = generate_mask(batch_tries, self.policy.model.config.vocab_size)
            token_mask = token_mask.to(self.device)

            with torch.no_grad(), autocast(self.policy.device.type):
                logits, values = self._safe_forward(batch_input_ids, batch_attention_mask)
            
            last_token_logits = logits[:, -1, :] + token_mask.to(dtype=logits.dtype)
            next_token_ids, chosen_log_probs = self._temperature_sampling(last_token_logits, temperature)

            for i, (token_id, log_prob, valid_token_ids) in enumerate(zip(next_token_ids, chosen_log_probs, all_valid_tokens_batch)):
                original_turn_idx = active_turn_indices[i]
                valid_per_step[original_turn_idx].append(valid_token_ids)
                path = tries[original_turn_idx].update_head(token_id)
                
                step_data[original_turn_idx].append((path, log_prob.item(), values[i].item()))
                
                turn_sequences[original_turn_idx].extend(path)
                
                if tries[original_turn_idx].head.value == self.eos_token_id:
                    turn_complete_mask[original_turn_idx] = True

        completed_actions_text = [
             self.tokenizer_helper.tokenizer.decode(sum((step[0] for step in data), [])).replace(self.tokenizer_helper.tokenizer.eos_token, "").strip()
             for data in step_data
        ]
        
        turn_log_probs = [sum(step[1] for step in data) for data in step_data]

        return GenerationResult(
            completed_actions=completed_actions_text,
            step_data=step_data,
            turn_log_probs=turn_log_probs,
            valid_per_step=valid_per_step
        )

    def _safe_forward(self, batch_input_ids, batch_attention_mask):
        try:
            torch.cuda.empty_cache()
            return self.policy.forward(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = batch_input_ids.shape[0]
            if batch_size <= 1:
                print(f"ERROR: CUDA OOM with a single sequence of length {batch_input_ids.shape[1]}.")
                logits_dummy = torch.zeros((1, batch_input_ids.shape[1], self.policy.model.config.vocab_size), dtype=torch.float32)
                values_dummy = torch.zeros((1, 1), dtype=torch.float32)
                return logits_dummy, values_dummy
            print(f"WARNING: CUDA OOM detected with batch size {batch_size}. Splitting batch.")
            mid = batch_size // 2
            logits1, values1 = self._safe_forward(batch_input_ids[:mid], batch_attention_mask[:mid])
            logits2, values2 = self._safe_forward(batch_input_ids[mid:], batch_attention_mask[mid:])
            return torch.cat((logits1, logits2), dim=0), torch.cat((values1, values2), dim=0)

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


class ExperienceRoller:
    """Orchestrates running environments and collecting experiences."""
    def __init__(self, policy: LLMPolicy, config: PPOConfig, device: torch.device, tokenizer_helper: TokenizerHelper):
        self.policy = policy
        self.config = config
        self.device = device
        self.tokenizer_helper = tokenizer_helper
        self.action_generator = ActionGenerator(policy, tokenizer_helper, device)
        self.start_token_id = self.tokenizer_helper.tokenizer.encode(">", add_special_tokens=False)[0]

        self.train_envs = [
            TextWorldEnvironment(config=self.config, is_eval_env=False, game_id=i) 
            for i in range(self.config.num_envs)
        ]
        
        self.eval_envs = [
            TextWorldEnvironment(config=self.config, is_eval_env=True, game_id=i) 
            for i in range(self.config.num_eval_episodes)
        ]

    def run(self, is_eval_mode: bool, temperature: float, num_episodes: Optional[int] = None, envs: Optional[List[TextWorldEnvironment]] = None) -> Tuple[List[Dict], List[EpisodeStats]]:
        """
        Runs a full rollout generation for either training or evaluation using internal envs.
        """
        if envs is None:
            if is_eval_mode:
                environments_to_use = self.eval_envs
            else:
                environments_to_use = self.train_envs
        else:
            environments_to_use = envs

        env_manager = ParallelEnvManager(
            config=self.config, 
            is_eval_env=is_eval_mode, 
            existing_envs=environments_to_use
        )
        total_episodes = env_manager.num_episodes

        rollout_buffer = []
        all_episode_stats = []
        pbar_desc = "Evaluating" if is_eval_mode else "Collecting Experience"
        with tqdm(total=total_episodes, desc=pbar_desc) as pbar:
            while not env_manager.is_done():
                active_indices = env_manager.get_active_indices()
                active_observations = env_manager.get_active_observations()
                valid_actions = env_manager.get_valid_actions()

                tries = self._prepare_tries(valid_actions)
                initial_prompts = [format_prompt(obs) for obs in active_observations]
                
                temp_to_use = 0.0 if is_eval_mode else temperature
                generation_result = self.action_generator.generate(initial_prompts, tries, temp_to_use)
                
                buffer_indices_map = {}
                if not is_eval_mode:
                    buffer_indices_map = self._store_step_experiences(rollout_buffer, generation_result, active_indices, active_observations)
                
                completed_this_turn, rewards, dones, truncateds = env_manager.step(generation_result.completed_actions, generation_result.turn_log_probs)

                if not is_eval_mode:
                    self._update_turn_in_buffer(rollout_buffer, buffer_indices_map, rewards, dones, truncateds)
                
                if completed_this_turn:
                    all_episode_stats.extend(completed_this_turn)
                    pbar.update(len(completed_this_turn))

        env_manager.close()
            
        return rollout_buffer, all_episode_stats
    
    def close(self):
        """Closes all managed environments."""
        print("Closing all environments managed by ExperienceRoller.")
        for env in self.train_envs:
            env.close()
        for env in self.eval_envs:
            env.close()

    def _prepare_tries(self, available_actions: List[List[str]]) -> List[Trie]:
        tries = []
        tokenized_actions_lists = [tokenize_actions_for_trie(act, self.tokenizer_helper.tokenizer) for act in available_actions]
        for token_ids in tokenized_actions_lists:
            trie = Trie(self.start_token_id)
            for action_tokens in token_ids:
                trie.insert(action_tokens)
            tries.append(trie)
        return tries

    def _store_step_experiences(self, buffer: List[Dict], result: GenerationResult, active_indices: List[int], active_observations: List[str]) -> Dict[int, List[int]]:
        """
        Stores token-level experiences in the buffer, interleaving steps from
        parallel environments to match the original implementation's logic.
        """
        indices_map = {idx: [] for idx in active_indices}
        num_active_envs = len(active_indices)
        
        # Track the partial command string for each active environment
        partial_command_strs = [""] * num_active_envs

        # Find the maximum number of generation steps in this turn
        max_steps = max(len(data) for data in result.step_data) if result.step_data else 0

        # Loop token-step by token-step (interleaving)
        for k in range(max_steps):
            # Loop through each active environment for the current token step
            for i in range(num_active_envs):
                # Check if this environment has a k-th generation step
                if k < len(result.step_data[i]):
                    original_env_idx = active_indices[i]
                    
                    # Get the data for this specific step
                    path, log_prob, value = result.step_data[i][k]
                    
                    # The state is the observation + the command *before* this step's token
                    state_for_log = (active_observations[i], partial_command_strs[i])
                    
                    entry = {
                        "env_idx": original_env_idx,
                        "state": state_for_log,
                        "action": path,
                        "old_logprob": log_prob,
                        "value": value,
                        "reward": 0.0,
                        "finished": False,
                        "truncated": False, 
                        "valid_token_ids": result.valid_per_step[i][k],
                    }
                    buffer.append(entry)
                    indices_map[original_env_idx].append(len(buffer) - 1)
                    
                    # Update the partial command string *after* logging the state
                    partial_command_strs[i] += self.tokenizer_helper.tokenizer.decode(path)
        
        return indices_map

    def _update_turn_in_buffer(self, buffer: List[Dict], indices: Dict[int, List[int]], rewards: Dict[int, float], dones: Dict[int, bool], truncateds: Dict[int, bool]):
        for env_idx, buffer_indices in indices.items():
            if not buffer_indices:
                continue

            reward = rewards.get(env_idx, 0.0)
            done = dones.get(env_idx, False)
            truncated = truncateds.get(env_idx, False)
            
            reward_per_step = reward / len(buffer_indices) if len(buffer_indices) > 0 else 0.0
            for buf_idx in buffer_indices:
                buffer[buf_idx]['reward'] = reward_per_step
            
            last_step_idx = buffer_indices[-1]
            buffer[last_step_idx]['finished'] = done
            buffer[last_step_idx]['truncated'] = truncated