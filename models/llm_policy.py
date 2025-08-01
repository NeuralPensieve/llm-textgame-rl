import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import logging
from torch.amp import autocast

from config import PPOConfig


class LLMPolicy(nn.Module):
    """Improved LLM-based policy for TextWorld with dual action evaluation methods"""
    
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            max_memory={0: "8GB"}
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Add value head
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Move value head to model device
        self.value_head = self.value_head.to(self.model.device)
        
        # Always cache target tokens, as 'high' is needed for value scoring
        self._cache_target_tokens()
    
    def _cache_target_tokens(self):
        """Cache target token IDs for 'helpful' and 'high' scoring"""
        self.target_tokens = {}
        
        # Action evaluation token
        helpful_token_id = self.tokenizer.encode(" helpful", add_special_tokens=False)
        if helpful_token_id:
            self.target_tokens["helpful"] = helpful_token_id[-1]
        else:
            raise ValueError("Could not tokenize 'helpful'")
        
        # Value evaluation token
        high_token_id = self.tokenizer.encode(" high", add_special_tokens=False)
        if high_token_id:
            self.target_tokens["high"] = high_token_id[-1]
        else:
            raise ValueError("Could not tokenize 'high'")
        
        self.logger.info(f"Cached target tokens: {self.target_tokens}")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through model with FP16"""
        try:
            with autocast(self.model.device.type):
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                # Get hidden states from last layer
                hidden_states = outputs.hidden_states[-1]
                
                # Use last token's representation for value estimation
                last_hidden = hidden_states[:, -1, :]
                value = self.value_head(last_hidden)
        
        except Exception as e:
            self.logger.error(f"Model forward pass failed: {e}")
            raise
        
        return outputs.logits, value
    
    def create_prompts(self, states: List[str], action_lists: List[List[str]]):
        """Create prompts for action evaluation based on scoring method"""
        action_prompts = []
        value_prompts = []
        metadata = []
        
        for i, (state, actions) in enumerate(zip(states, action_lists)):
            for action in actions:
                if self.config.use_action_token_scoring:
                    prompt = f"In game state: {state}, best action is {action}"
                else:
                    prompt = f"In this text adventure:\n{state}\n\nConsidering action: {action}\nThis action is helpful"
                action_prompts.append(prompt)
                metadata.append((i, action.strip()))
            
            value_prompt = f"Game state:\n{state}\n\nThe value of this state is high"
            value_prompts.append(value_prompt)
        
        return action_prompts, value_prompts, metadata
    
    def tokenize_prompts(self, prompts: List[str]):
        """Tokenize prompts efficiently with caching"""
        self.tokenizer.padding_side = "left"

        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_attention_mask=True
        )
    
    def compute_action_scores(self, logits: torch.Tensor, input_ids: torch.Tensor, metadata: List[Tuple], num_states: int, action_prompts: List[str]) -> List[List[torch.Tensor]]:
        """Compute action scores based on scoring method"""
        env_action_logprobs = [[] for _ in range(num_states)]
        
        if self.config.use_action_token_scoring:
            logprobs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
            
            for idx, ((env_idx, action), prompt) in enumerate(zip(metadata, action_prompts)):
                if not action.strip():
                    raise ValueError(f"Empty or whitespace action for env_idx {env_idx}: {action}")
                    
                
                # Tokenize action with leading space for BPE-based tokenizers
                action_clean = action.strip()
                action_tokens = self.tokenizer.encode(' ' + action_clean, add_special_tokens=False)
                
                n = len(action_tokens)
                prompt_tokens = input_ids[idx][-n:].cpu().tolist()
                
                # Validate token alignment
                if prompt_tokens != action_tokens:
                    prompt_token_strings = self.tokenizer.convert_ids_to_tokens(prompt_tokens)
                    action_token_strings = self.tokenizer.convert_ids_to_tokens(action_tokens)
                    raise ValueError(
                        f"Token mismatch for action '{action_clean}' (env_idx {env_idx}). "
                        f"Expected tokens: {action_tokens} ({action_token_strings}), "
                        f"Got prompt tokens: {prompt_tokens} ({prompt_token_strings})"
                    )
                
                # Compute score
                logprobs_i = logprobs[idx, -n:]  # shape: [n, vocab_size]
                if logprobs_i.shape[0] < n:
                    raise ValueError(f"Insufficient tokens for action: {action_clean}. Expected {n}, got {logprobs_i.shape[0]}")
                
                actions_tensor = torch.tensor(action_tokens, device=logits.device)  # shape: [n]
                selected = logprobs_i.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # shape: [n]
                score = selected.mean()  # Keep as tensor
                env_action_logprobs[env_idx].append(score)
        else:
            logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
            helpful_token_id = self.target_tokens["helpful"]
            for (env_idx, action), logprob_dist in zip(metadata, logprobs):
                if not action.strip():
                    self.logger.warning(f"Empty or whitespace action for env_idx {env_idx}: {action}")
                    env_action_logprobs[env_idx].append(torch.tensor(-10.0, device=logits.device))
                    continue
                score = logprob_dist[helpful_token_id]
                env_action_logprobs[env_idx].append(score)
        
        # Ensure non-empty lists
        for env_idx in range(num_states):
            if not env_action_logprobs[env_idx]:
                raise ValueError(f"No action scores computed for env_idx {env_idx}. Using fallback score.")
        
        return env_action_logprobs
    
    def compute_value_scores(self, logprobs: torch.Tensor) -> torch.Tensor:
        """Compute value scores using 'high' token probability"""
        high_token_id = self.target_tokens["high"]
        value_scores = logprobs[:, high_token_id]
        return value_scores
    
    def evaluate_actions(self, states: List[str], action_lists: List[List[str]]) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
        """Evaluate actions with gradients (for training)"""
        self.train()
        
        action_prompts, value_prompts, metadata = self.create_prompts(states, action_lists)
        action_inputs = self.tokenize_prompts(action_prompts)
        value_inputs = self.tokenize_prompts(value_prompts)
        
        action_inputs = {k: v.to(self.model.device) for k, v in action_inputs.items()}
        value_inputs = {k: v.to(self.model.device) for k, v in value_inputs.items()}
        
        with autocast(self.model.device.type):
            action_logits, _ = self.forward(**action_inputs)
            value_logits, _ = self.forward(**value_inputs)
            value_logprobs = F.log_softmax(value_logits[:, -1, :], dim=-1)
        
        env_action_logprobs = self.compute_action_scores(action_logits, action_inputs['input_ids'], metadata, len(states), action_prompts)
        values = self.compute_value_scores(value_logprobs)
        
        return env_action_logprobs, values
    
    def evaluate_for_rollout(self, states: List[str], action_lists: List[List[str]]) -> Tuple[List[List[float]], List[float]]:
        """Evaluate actions without gradients (for rollout collection)"""
        self.eval()
        env_action_logprobs = []
        values_list = []
        
        for state, actions in zip(states, action_lists):
            prompt_actions = []
            for action in actions:
                if self.config.use_action_token_scoring:
                    prompt = f"In game state: {state}, best action is {action}"
                else:
                    prompt = f"In this text adventure:\n{state}\n\nConsidering action: {action}\nThis action is helpful"
                prompt_actions.append(prompt)
            
            inputs = self.tokenize_prompts(prompt_actions)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad(), autocast(self.model.device.type):
                logits, _ = self.forward(**inputs)
                if self.config.use_action_token_scoring:
                    action_scores = []
                    logprobs = F.log_softmax(logits, dim=-1)  # [action_batch, seq_len, vocab_size]

                    # We add a space before each action to ensure tokenization matches
                    actions = [f" {action}" for action in actions]
                    actions_tokens = [self.tokenizer.encode(action, add_special_tokens=False) for action in actions]

                    action_batch_size = len(actions_tokens)
                    input_ids = inputs['input_ids']  # [batch, seq_len]

                    for i in range(action_batch_size):
                        n = len(actions_tokens[i])

                        # Get last n tokens from prompt's input_ids
                        prompt_tokens = input_ids[i][-n:].cpu().tolist()
                        expected_tokens = actions_tokens[i]

                        # Validate token alignment
                        if prompt_tokens != expected_tokens:
                            prompt_token_strings = self.tokenizer.convert_ids_to_tokens(prompt_tokens)
                            expected_token_strings = self.tokenizer.convert_ids_to_tokens(expected_tokens)
                            self.logger.error(
                                f"Token mismatch for action '{actions[i]}'. "
                                f"Expected tokens: {expected_tokens} ({expected_token_strings}), "
                                f"Got prompt tokens: {prompt_tokens} ({prompt_token_strings})"
                            )
                            raise ValueError(
                                f"Action tokens do not match prompt tokens for action '{actions[i]}'. "
                                f"Expected: {expected_token_strings}, Got: {prompt_token_strings}. "
                                "Check tokenizer compatibility or prompt structure."
                            )

                        # Proceed with scoring
                        logprobs_i = logprobs[i, -n:]  # shape: [n, vocab_size]
                        if logprobs_i.shape[0] < n:
                            raise ValueError(f"Insufficient tokens for action: {actions[i]}. Expected {n}, got {logprobs_i.shape[0]}")
                            

                        actions_tensor = torch.tensor(actions_tokens[i], device=logprobs.device)  # shape: [n]
                        selected = logprobs_i.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # shape: [n]
                        avg = selected.mean().cpu().item()
                        action_scores.append(avg)
                                
                else:
                    logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
                    helpful_token_id = self.target_tokens["helpful"]
                    action_scores = [logprob[helpful_token_id].cpu().item() for logprob in logprobs]
                
            env_action_logprobs.append(action_scores)


            
            
            del inputs, logits, logprobs
            torch.cuda.empty_cache()
            
            value_prompt = f"Game state:\n{state}\n\nThe value of this state is high"
            value_inputs = self.tokenize_prompts([value_prompt])
            value_inputs = {k: v.to(self.model.device) for k, v in value_inputs.items()}
            
            with torch.no_grad(), autocast(self.model.device.type):
                logits, _ = self.forward(**value_inputs)
                logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
                high_token_id = self.target_tokens["high"]
                value_score = logprobs[0][high_token_id].cpu().item()
                values_list.append(value_score)
            
            del value_inputs, logits, logprobs
            torch.cuda.empty_cache()
        
        return env_action_logprobs, values_list
    
    def get_kl_loss(self, new_logits: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss to prevent catastrophic forgetting"""
        with autocast(self.model.device.type):
            old_probs = F.softmax(old_logits.detach(), dim=-1)
            kl_loss = F.kl_div(
                F.log_softmax(new_logits, dim=-1),
                old_probs,
                reduction='batchmean'
            )
        return kl_loss
    
    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text response (for testing/debugging)"""
        self.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad(), autocast(self.model.device.type):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def get_device(self):
        """Get model device"""
        return self.model.device
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.model.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Clear any cached key-value pairs
        if hasattr(self.model, 'past_key_values'):
            self.model.past_key_values = None
    
    def get_separate_parameter_groups(self):
        """Get parameter groups with different learning rates"""
        return [
            {'params': self.model.parameters(), 'lr': 1e-5, 'name': 'pretrained'},
            {'params': self.value_head.parameters(), 'lr': 3e-4, 'name': 'value_head'}
        ]