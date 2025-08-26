import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import List, Tuple
import logging
from torch.amp import autocast
from peft import LoraConfig, get_peft_model, TaskType

from config import PPOConfig
from helper import format_prompt, TokenizerHelper

class LLMPolicy(nn.Module):
    def __init__(self, config: PPOConfig, tokenizer_helper: TokenizerHelper, device: torch.device):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tokenizer_helper = tokenizer_helper
        self.device = device

        # Load the model configuration, ensuring use_cache=False for training efficiency.
        model_config = AutoConfig.from_pretrained(config.model_name, use_cache=False)

        model_kwargs = {
            "config": model_config,
            "torch_dtype": torch.float16 if config.use_fp16 else torch.float32,
            }

        if "qwen" in config.model_name.lower():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load the model by unpacking the kwargs dictionary
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        ).to(self.device)
        
        # Enable gradient checkpointing on the main model to save memory during training.
        base_model.gradient_checkpointing_enable()

        self.model = base_model # Temporarily assign base_model

        # 3. CREATE REFERENCE MODEL (IF NEEDED) BEFORE APPLYING LORA
        if config.use_kl_penalty:
            self.logger.info("Creating reference model for KL penalty...")
            # The reference model should be the original, un-adapted model
            self.reference_model = copy.deepcopy(self.model)
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            if hasattr(self.reference_model, 'gradient_checkpointing_disable'):
                self.reference_model.gradient_checkpointing_disable()
            self.logger.info("Reference model created successfully.")

        # 4. APPLY LORA ADAPTERS IF ENABLED
        if self.config.lora_enabled:
            self.logger.info("Applying LoRA adapters to the model...")
            model_name_lower = self.config.model_name.lower()
            if "qwen" in model_name_lower:
                target_modules = ["q_proj", "v_proj"]
            elif "gpt2" in model_name_lower or "dialogpt" in model_name_lower:
                target_modules = ["c_attn", "c_proj"]
            else:
                raise ValueError(
                    f"LoRA target modules are not defined for model: {self.config.model_name}. "
                    "Please add the target module names in llm_policy.py."
                )
            # Define LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            # Wrap the model with LoRA adapters
            self.model = get_peft_model(self.model, lora_config)
            self.logger.info("LoRA applied successfully. Trainable parameters:")
            self.model.print_trainable_parameters()
        
        # The value head predicts the value of a state (V(s)).
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Add normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        ).to(self.device)

        # Major improvement: Initialize value head to predict near zero
        with torch.no_grad():
            for layer in self.value_head:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)  # Very small weights
                    nn.init.constant_(layer.bias, 0.0)
            
            # Set final layer bias to expected value
            expected_value = -1 * self.config.step_penalty
            self.value_head[-1].bias.data.fill_(expected_value)

        model_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        value_head_params = sum(p.numel() for p in self.value_head.parameters())
        self.logger.info(f"Policy model total parameters: {model_params / 1e6:.2f}M")
        self.logger.info(f"Policy model trainable parameters: {trainable_params / 1e3:.2f}K")
        self.logger.info(f"Value head parameters: {value_head_params / 1e3:.2f}K")


    def forward(self, input_ids, attention_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass on the model. This is used by the ExperienceRoller
        for generating actions during rollouts.
        """
        # autocast is still important for operations outside the main model, like the value head
        with autocast(self.device.type):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            if self.config.disable_value_function:  # Add this config flag
                # Return zeros for values
                batch_size = input_ids.shape[0]
                value = torch.zeros((batch_size, 1), device=self.device)
            else:
                hidden_states = outputs.hidden_states[-1] 
                last_hidden = hidden_states[:, -1, :]
                value = self.value_head(last_hidden.detach())  # To prevent backpropagation through value head

        return outputs.logits, value

    def evaluate_tokens(
        self, 
        composite_states: List[Tuple[str, str]], 
        chosen_tokens: List[int],
        valid_token_ids: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-evaluates the log probability of a batch of token-level actions
        for the PPO loss calculation.
        """
        prompts = [f"{format_prompt(obs)}{partial}" for obs, partial in composite_states]
        
        padded_batch = self.tokenizer_helper.tokenize_prompts(prompts)
        input_ids = padded_batch["input_ids"].to(self.device)
        attention_mask = padded_batch["attention_mask"].to(self.device)
        
        chosen_tokens_tensor = torch.LongTensor(chosen_tokens).to(self.device).unsqueeze(1)

        logits, values = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        last_token_logits = logits[:, -1, :].clone()

        # Create a boolean mask for valid tokens
        batch_size, vocab_size = last_token_logits.shape
        valid_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=self.device)
        
        for i, valid_ids in enumerate(valid_token_ids):
            valid_mask[i, valid_ids] = True
        
        # Apply -inf to invalid positions
        masked_logits = last_token_logits.masked_fill(~valid_mask, float('-inf'))
        
        # Compute probabilities and log probabilities
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        
        # Key fix: Set log_probs to 0 where probs are 0 to avoid 0 * -inf = NaN
        # This is mathematically sound since lim(p→0) p*log(p) = 0
        safe_log_probs = torch.where(valid_mask, log_probs, torch.zeros_like(log_probs))
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = -(probs * safe_log_probs).sum(dim=-1)
        
        # Get chosen log probabilities
        chosen_tokens_tensor = torch.LongTensor(chosen_tokens).to(self.device).unsqueeze(1)
        chosen_log_probs = torch.gather(log_probs, 1, chosen_tokens_tensor).squeeze(1)

        # Get the sequence length from the padded input_ids tensor
        batch_seq_len = input_ids.shape[1]

        # Return the sequence length along with the other values
        return chosen_log_probs, values.squeeze(-1), entropy, batch_seq_len

    def get_separate_parameter_groups(self):
        """Get parameter groups with different learning rates for the optimizer."""

        # 5. CRITICAL: ONLY PASS TRAINABLE (LORA) PARAMETERS TO THE OPTIMIZER
        if self.config.lora_enabled:
            # The optimizer should only see the trainable adapter parameters and the value head
            model_trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            return [
                {"params": model_trainable_params, "lr": self.config.learning_rate, "name": "lora_adapters"},
                {"params": self.value_head.parameters(), "lr": self.config.learning_rate_value_head, "name": "value_head"},
            ]
        else:
            # Original behavior if LoRA is disabled
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