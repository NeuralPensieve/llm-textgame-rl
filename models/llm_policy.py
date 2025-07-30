import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import logging

from config import PPOConfig


class LLMPolicy(nn.Module):
    """Improved LLM-based policy for TextWorld with better action evaluation"""
    
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
        
        # Cache target token IDs for efficient scoring
        self._cache_target_tokens()
    
    def _cache_target_tokens(self):
        """Cache target token IDs for efficient scoring"""
        self.target_tokens = {}
        
        # Action evaluation token
        helpful_token_id = self.tokenizer.encode(" helpful", add_special_tokens=False)
        if helpful_token_id:
            self.target_tokens["helpful"] = helpful_token_id[-1]
        else:
            self.logger.warning("Could not tokenize 'helpful'")
            self.target_tokens["helpful"] = self.tokenizer.unk_token_id
        
        # Value evaluation token
        high_token_id = self.tokenizer.encode(" high", add_special_tokens=False)
        if high_token_id:
            self.target_tokens["high"] = high_token_id[-1]
        else:
            self.logger.warning("Could not tokenize 'high'")
            self.target_tokens["high"] = self.tokenizer.unk_token_id
        
        self.logger.info(f"Cached target tokens: {self.target_tokens}")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through model"""
        try:
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
        except Exception as e:
            self.logger.error(f"Model forward pass failed: {e}")
            raise
        
        # Get hidden states from last layer
        hidden_states = outputs.hidden_states[-1]
        
        # Use last token's representation for value estimation
        last_hidden = hidden_states[:, -1, :]
        value = self.value_head(last_hidden)
        
        return outputs.logits, value
    
    def create_prompts(self, states: List[str], action_lists: List[List[str]]):
        """Create prompts for action evaluation"""
        action_prompts = []
        value_prompts = []
        metadata = []
        
        for i, (state, actions) in enumerate(zip(states, action_lists)):
            # Single action evaluation per action
            for action in actions:
                prompt = f"In this text adventure:\n{state}\n\nConsidering action: {action}\nThis action is helpful"
                action_prompts.append(prompt)
                metadata.append((i, action))
            
            # Single value prompt per state
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
    
    def compute_action_scores(self, logprobs: torch.Tensor, metadata: List[Tuple], num_states: int) -> List[List[torch.Tensor]]:
        """Compute action scores using 'helpful' token probability"""
        env_action_logprobs = [[] for _ in range(num_states)]
        
        helpful_token_id = self.target_tokens["helpful"]
        
        for (env_idx, action), logprob_dist in zip(metadata, logprobs):
            score = logprob_dist[helpful_token_id]
            env_action_logprobs[env_idx].append(score)
        
        return env_action_logprobs
    
    def compute_value_scores(self, logprobs: torch.Tensor) -> torch.Tensor:
        """Compute value scores using 'high' token probability"""
        high_token_id = self.target_tokens["high"]
        value_scores = logprobs[:, high_token_id]
        return value_scores
    
    def evaluate_actions(self, states: List[str], action_lists: List[List[str]]) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
        """Evaluate actions with gradients (for training)"""
        self.train()
        
        # Create prompts
        action_prompts, value_prompts, metadata = self.create_prompts(states, action_lists)
        # Tokenize
        action_inputs = self.tokenize_prompts(action_prompts)
        value_inputs = self.tokenize_prompts(value_prompts)
        
        # Move to device
        action_inputs = {k: v.to(self.model.device) for k, v in action_inputs.items()}
        value_inputs = {k: v.to(self.model.device) for k, v in value_inputs.items()}
        
        # Forward pass for actions
        action_logits, _ = self.forward(**action_inputs)
        action_logprobs = F.log_softmax(action_logits[:, -1, :], dim=-1)

        # Forward pass for values
        value_logits, _ = self.forward(**value_inputs)
        value_logprobs = F.log_softmax(value_logits[:, -1, :], dim=-1)
        
        # Compute scores using improved methods
        env_action_logprobs = self.compute_action_scores(action_logprobs, metadata, len(states))
        values = self.compute_value_scores(value_logprobs)
        
        return env_action_logprobs, values
    
    def evaluate_for_rollout(self, states: List[str], action_lists: List[List[str]]) -> Tuple[List[List[float]], List[float]]:
        """Evaluate actions without gradients (for rollout collection) - Single criterion"""
        self.eval()
        env_action_logprobs = [[] for _ in range(len(states))]
        values_list = []
        
        # Process each state individually to save memory
        for i, (state, actions) in enumerate(zip(states, action_lists)):
            # Create action prompts
            action_prompts = []
            for action in actions:
                prompt = f"In this text adventure:\n{state}\n\nConsidering action: {action}\nThis action is helpful"
                action_prompts.append(prompt)
            
            if action_prompts:
                # Tokenize and evaluate actions
                inputs = self.tokenize_prompts(action_prompts)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits, _ = self.forward(**inputs)
                    logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
                    
                    # Get scores using helpful token
                    helpful_token_id = self.target_tokens["helpful"]
                    action_scores = [logprob[helpful_token_id].cpu().item() for logprob in logprobs]
                    env_action_logprobs[i] = action_scores
                
                # Clean up
                del inputs, logits, logprobs
                torch.cuda.empty_cache()
            
            # Evaluate value
            value_prompt = f"Game state:\n{state}\n\nThe value of this state is high"
            value_inputs = self.tokenize_prompts([value_prompt])
            value_inputs = {k: v.to(self.model.device) for k, v in value_inputs.items()}
            
            with torch.no_grad():
                logits, _ = self.forward(**value_inputs)
                logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Get value using high token
                high_token_id = self.target_tokens["high"]
                value_score = logprobs[0][high_token_id].cpu().item()
                values_list.append(value_score)
            
            # Clean up
            del value_inputs, logits, logprobs
            torch.cuda.empty_cache()
        
        return env_action_logprobs, values_list
    
    def get_kl_loss(self, new_logits: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss to prevent catastrophic forgetting"""
        new_probs = F.softmax(new_logits, dim=-1)
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
        
        with torch.no_grad():
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