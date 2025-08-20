import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import List, Tuple
import logging
from torch.amp import autocast

from config import PPOConfig
from helper import format_prompt, TokenizerHelper

class LLMPolicy(nn.Module):
    def __init__(self, config: PPOConfig, tokenizer_helper: TokenizerHelper, device: torch.device):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tokenizer_helper = tokenizer_helper

        # Load the model configuration, ensuring use_cache=False for training efficiency.
        model_config = AutoConfig.from_pretrained(config.model_name, use_cache=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
        ).to(device)

        self.device = device
        
        # Enable gradient checkpointing on the main model to save memory during training.
        self.model.gradient_checkpointing_enable()

        # The value head predicts the value of a state (V(s)).
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        ).to(self.device)

        model_params = sum(p.numel() for p in self.model.parameters())
        value_head_params = sum(p.numel() for p in self.value_head.parameters())
        self.logger.info(f"Policy model parameters: {model_params / 1e6:.2f}M")
        self.logger.info(f"Value head parameters: {value_head_params / 1e3:.2f}K")

        if config.use_kl_penalty:
            self.logger.info("Creating reference model for KL penalty...")
            self.reference_model = copy.deepcopy(self.model)
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            if hasattr(self.reference_model, 'gradient_checkpointing_disable'):
                self.reference_model.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass on the model. This is used by the ExperienceRoller
        for generating actions during rollouts.
        """
        with autocast(self.device.type):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            last_hidden = hidden_states[:, -1, :]
            value = self.value_head(last_hidden)

        return outputs.logits, value

    def evaluate_tokens(
        self, 
        composite_states: List[Tuple[str, str]], 
        chosen_tokens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-evaluates the log probability of a batch of token-level actions
        for the PPO loss calculation.
        """
        # Construct prompts from the (game_observation, partial_command) tuples
        prompts = [f"{format_prompt(obs)}{partial}" for obs, partial in composite_states]
        
        # Tokenize the batch of prompts
        padded_batch = self.tokenizer_helper.tokenize_prompts(prompts)
        input_ids = padded_batch["input_ids"].to(self.device)
        attention_mask = padded_batch["attention_mask"].to(self.device)
        
        chosen_tokens_tensor = torch.LongTensor(chosen_tokens).to(self.device).unsqueeze(1)

        # Perform a forward pass to get new logits and values under the current policy
        logits, values = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        
        last_token_logits = logits[:, -1, :]
        log_probs = F.log_softmax(last_token_logits, dim=-1)
        
        # Gather the logprobs of the specific tokens that were actually chosen
        chosen_log_probs = torch.gather(log_probs, 1, chosen_tokens_tensor).squeeze(1)

        return chosen_log_probs, values.squeeze(-1)

    def get_separate_parameter_groups(self):
        """Get parameter groups with different learning rates for the optimizer."""
        return [
            {"params": self.model.parameters(), "lr": self.config.learning_rate, "name": "pretrained"},
            {"params": self.value_head.parameters(), "lr": self.config.learning_rate_value_head, "name": "value_head"},
        ]
    
    def get_reference_token_logprobs(
        self, 
        composite_states: List[Tuple[str, str]], 
        chosen_tokens: List[int]
    ) -> torch.Tensor:
        """
        Gets the log probability of chosen tokens from the frozen reference model.
        Used for calculating KL divergence penalty.
        """
        if not hasattr(self, 'reference_model'):
            raise ValueError("KL penalty is enabled, but no reference model was created.")

        prompts = [f"{obs.strip()}\n>{partial}" for obs, partial in composite_states]
        padded_batch = self.tokenizer_helper.tokenize_prompts(prompts)
        input_ids = padded_batch["input_ids"].to(self.device)
        attention_mask = padded_batch["attention_mask"].to(self.device)
        chosen_tokens_tensor = torch.LongTensor(chosen_tokens).to(self.device).unsqueeze(1)

        with torch.no_grad():
            ref_outputs = self.reference_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits[:, -1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_chosen_log_probs = torch.gather(ref_log_probs, 1, chosen_tokens_tensor).squeeze(1)
            
        return ref_chosen_log_probs
    
    def compute_kl_divergence(
        self, 
        current_logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes KL divergence between the current and reference token log probabilities.
        KL(current || reference) = Σ P_current(x) * log(P_current(x) / P_reference(x))
        """
        kl_div = (torch.exp(current_logprobs) * (current_logprobs - reference_logprobs)).sum()
        return kl_div / len(current_logprobs) # Return mean KL
