import re
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Tuple
import logging
from torch.amp import autocast

from config import PPOConfig
from config import PromptManager


class LLMPolicy(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.prompt_manager = PromptManager(config.scoring_method)
        self.logger = logging.getLogger(__name__)

        # Set up the tokenizer, adding a padding token if it's missing.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model configuration, ensuring use_cache=False for training efficiency.
        model_config = AutoConfig.from_pretrained(config.model_name, use_cache=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
            # device_map="auto",
            # max_memory={0: "8GB"},
        )
        
        # Enable gradient checkpointing on the main model to save memory during training.
        self.model.gradient_checkpointing_enable()

        # The value head predicts the value of a state (V(s)).
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        ).to(self.model.device) # Move the head to the same device as the model.

        # Log the number of parameters for both the main model and the value head.
        model_params = sum(p.numel() for p in self.model.parameters())
        value_head_params = sum(p.numel() for p in self.value_head.parameters())
        self.logger.info(f"Policy model parameters: {model_params / 1e6:.2f}M")
        self.logger.info(f"Value head parameters: {value_head_params / 1e3:.2f}K")

        if config.use_kl_penalty:
            # Load a second, separate instance of the model to act as a fixed reference.
            self.logger.info("Creating reference model by loading state_dict...")
            
            self.reference_model = copy.deepcopy(self.model)

            # Freeze the reference model's parameters and set it to evaluation mode.
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            
            # Freeze the reference model's parameters and set it to evaluation mode.
            # It should never be trained.
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            
            # Disable gradient checkpointing for the reference model for faster inference.
            if hasattr(self.reference_model, 'gradient_checkpointing_disable'):
                self.reference_model.gradient_checkpointing_disable()
        
        self._cache_target_tokens()

        # if config.use_kl_penalty:
        #     self.logger.info("Running model identity sanity check...")
        #     with torch.no_grad():
        #         # Set both models to eval mode for a deterministic comparison.
        #         self.model.eval()
                
        #         dummy_input = self.tokenizer("hello world", return_tensors="pt").to(self.model.device)
                
        #         # Perform a direct, identical forward pass on both base models.
        #         main_outputs = self.model(**dummy_input)
        #         ref_outputs = self.reference_model(**dummy_input)
                
        #         # Assert that their logits are close, accounting for potential dtype differences.
        #         assert torch.allclose(
        #             main_outputs.logits.float(), 
        #             ref_outputs.logits.float(), 
        #             atol=1e-3
        #         ), "Model and reference model outputs do not match after initialization!"
                
        #         self.logger.info("✅ Sanity check passed: Models are identical.")

        #         # Optionally convert the reference model to FP16 to save memory.
        #         if config.reference_fp16:
        #             self.reference_model = self.reference_model.half()
                
        #         # IMPORTANT: Set the main model back to train mode for PPO updates.
        #         self.model.train()

    def _cache_target_tokens(self):
        """Cache target token IDs for 'helpful' scoring"""
        self.target_tokens = {}

        # Action evaluation token
        helpful_token_id = self.tokenizer.encode(" helpful", add_special_tokens=False)
        if helpful_token_id:
            self.target_tokens["helpful"] = helpful_token_id[-1]
        else:
            raise ValueError("Could not tokenize 'helpful'")

        self.logger.info(f"Cached target tokens: {self.target_tokens}")

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through model with FP16"""
        with autocast(self.model.device.type):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Get hidden states from last layer
            hidden_states = outputs.hidden_states[-1]

            # Use last token's representation for value estimation
            last_hidden = hidden_states[:, -1, :]
            value = self.value_head(last_hidden)

        return outputs.logits, value


    def tokenize_prompts(self, prompts: List[str]):
        """
        Tokenizes prompts with intelligent truncation of the history string for oversized prompts.
        """
        self.tokenizer.padding_side = "left"
        max_len = self.config.max_length
        
        truncated_prompts = []
        for prompt in prompts:
            # 1. First, tokenize the whole prompt to see if it needs truncation
            input_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]

            # 2. If it's too long, start trimming the history
            while len(input_ids) > max_len:
                # Find the history string using regex
                match = re.search(r"HISTORY: (.*?)\nSTATE:", prompt, re.DOTALL)
                if not match:
                    # # If no history is found, we can't shorten it.
                    # # Fallback to simple truncation (shouldn't happen with your format).
                    # prompt = prompt[:int(len(prompt) * 0.9)] # Failsafe
                    # input_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
                    # continue
                    raise ValueError('State too long, and there is no HISTORY to cut')

                history_str = match.group(1).strip()
                
                # Split history into chunks and remove the oldest (the first one)
                history_chunks = [chunk.strip() for chunk in history_str.split('|')]
                
                if len(history_chunks) > 1:
                    # Remove the oldest history entry
                    new_history = " | ".join(history_chunks[1:])
                else:
                    # If only one history chunk left, we can't shorten it further.
                    # To prevent an infinite loop, we clear it.
                    new_history = "None"

                # Rebuild the prompt with the shorter history
                start_index = match.start(1)
                end_index = match.end(1)
                prompt = prompt[:start_index] + new_history + prompt[end_index:]

                # Re-tokenize to check the new length
                input_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            
            truncated_prompts.append(prompt)

        # 3. Now, tokenize the (potentially truncated) prompts together with padding
        padded = self.tokenizer(
            truncated_prompts,
            padding="longest",
            truncation=False, # We've already handled truncation manually
            return_attention_mask=True,
            return_tensors="pt",
        )

        return padded

    def compute_action_scores(
        self,
        logits: torch.Tensor,
        metadata: List[Tuple],
        num_states: int,
        temperature: float,
    ) -> List[torch.Tensor]:
        """Compute action scores based on scoring method"""
        env_action_logprobs = [[] for _ in range(num_states)]

        if self.prompt_manager.scoring_method == "action_token":
            for idx, (env_idx, action) in enumerate(metadata):
                action_tokens = self.tokenizer.encode(
                    " " + action.strip(), add_special_tokens=False
                )
                n = len(action_tokens)

                token_logits = []
                for i, token_id in enumerate(action_tokens):
                    pos = -(n - i)
                    logit = logits[idx, pos, token_id]
                    token_logits.append(logit)

                avg_score = torch.stack(token_logits).mean()
                env_action_logprobs[env_idx].append(avg_score)
        elif self.prompt_manager.scoring_method == "helpful":
            helpful_token_id = self.target_tokens["helpful"]
            for (env_idx, action), logit in zip(metadata, logits[:, -1, :]):
                score = logit[helpful_token_id]
                env_action_logprobs[env_idx].append(score)
        else:
            raise ValueError(f"Wrong scoring_method: {self.prompt_manager.scoring_method}")

        env_action_logprobs = [F.log_softmax(torch.stack(row) / temperature, dim=-1) for row in env_action_logprobs]

        return env_action_logprobs

    def evaluate_actions(
        self, states: List[str], action_lists: List[List[str]], temperature: float
    ) -> Tuple[List[List[torch.Tensor]], torch.Tensor, int]:
        """
        Evaluate actions with gradients (for training).
        
        Returns a tuple of (action_logprobs, values, sequence_length).
        """
        # self.model.train()

        # Build action prompts
        action_prompts = []
        metadata = []
        for i, (state, actions) in enumerate(zip(states, action_lists)):
            for action in actions:
                prompt = self.prompt_manager.get_action_prompt(state, actions, action)
                action_prompts.append(prompt)
                metadata.append((i, action.strip()))

        # --- TRACE POINT 1: Prompts ---
        if self.config.debug_mode:
            print(f"[TRACE-MAIN] Prompts Hash: {hash(tuple(action_prompts))}")

        # Get action scores
        action_inputs = self.tokenize_prompts(action_prompts)
        action_inputs = {k: v.to(self.model.device) for k, v in action_inputs.items()}

        # --- TRACE POINT 2: Tokenized Input Tensor ---
        if self.config.debug_mode:
            print(f"[TRACE-MAIN] Input Tensor Sum: {action_inputs['input_ids'].sum().item()}")

        sequence_length = action_inputs['input_ids'].shape[1]
        
        with autocast(self.model.device.type):
            action_outputs = self.model(**action_inputs)
            action_logits = action_outputs.logits

        # --- TRACE POINT 3: Raw Logits Tensor ---
        if self.config.debug_mode:
            print(f"[TRACE-MAIN] Raw Logits Sum: {action_logits.sum().item()}")

        if self.config.debug_mode:
            print(f"[evaluate_actions] Input shape: {action_inputs['input_ids'].shape}")
            print(f"[evaluate_actions] First prompt tokens (last 10): {action_inputs['input_ids'][0, -10:].tolist()}")

        env_action_logprobs = self.compute_action_scores(
            action_logits,
            metadata,
            len(states),
            temperature,
        )
        
        state_inputs = self.tokenize_prompts(states)  # Just tokenize states directly
        state_inputs = {k: v.to(self.model.device) for k, v in state_inputs.items()}
        
        with autocast(self.model.device.type):
            _, values = self.forward(**state_inputs)  # Use value head output
        
        return env_action_logprobs, values.squeeze(-1), sequence_length

    def evaluate_for_rollout(
        self, states: List[str], action_lists: List[List[str]]
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Evaluate actions without gradients, ensuring the model's training
        state is preserved.
        """
        # 1. Save the model's original training mode
        was_training = self.model.training
        
        try:
            # 2. Set the model to evaluation mode for this operation
            self.model.eval()

            # The rest of the function's logic remains the same
            env_action_scores = []

            for state, actions in zip(states, action_lists):
                action_prompts = [
                    self.prompt_manager.get_action_prompt(state, actions, action)
                    for action in actions
                ]
                
                inputs = self.tokenize_prompts(action_prompts)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad(), autocast(self.model.device.type):
                    logits, _ = self.forward(**inputs)
                    # This part is unchanged
                    if self.prompt_manager.scoring_method == "action_token":
                        action_scores = []
                        actions_tokens = [self.tokenizer.encode(f" {a}", add_special_tokens=False) for a in actions]
                        input_ids = inputs["input_ids"]
                        for i, tokens in enumerate(actions_tokens):
                            n = len(tokens)
                            logits_i = logits[i, -n:]
                            actions_tensor = torch.tensor(tokens, device=logits.device)
                            selected = logits_i.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                            action_scores.append(selected.mean())
                    elif self.prompt_manager.scoring_method == "helpful":
                        helpful_token_id = self.target_tokens["helpful"]
                        action_scores = [logit[helpful_token_id] for logit in logits[:, -1, :]]
                    
                    action_scores_tensor = torch.stack(action_scores)
                    env_action_scores.append(action_scores_tensor.cpu().tolist())

            state_inputs = self.tokenize_prompts(states)
            state_inputs = {k: v.to(self.model.device) for k, v in state_inputs.items()}
            
            with torch.no_grad(), autocast(self.model.device.type):
                _, values = self.forward(**state_inputs)
                values_list = values.squeeze(-1).cpu().tolist()
            
            return env_action_scores, values_list

        finally:
            # 3. CRITICAL: Restore the original mode, no matter what happens
            if was_training:
                self.model.train()

    def get_separate_parameter_groups(self):
        """Get parameter groups with different learning rates"""
        return [
            {"params": self.model.parameters(), "lr": self.config.learning_rate, "name": "pretrained"},
            {"params": self.value_head.parameters(), "lr": self.config.learning_rate_value_head, "name": "value_head"},
        ]
    
    def get_reference_action_scores(self, states: List[str], action_lists: List[List[str]], temperature: float) -> List[torch.Tensor]:
        """Get action scores from reference model using the single, shared scoring method."""

        with torch.no_grad():
            # Build prompts and metadata
            action_prompts = []
            metadata = []  # (state_idx, action_text)
            
            for i, (state, actions) in enumerate(zip(states, action_lists)):
                for action in actions:
                    prompt = self.prompt_manager.get_action_prompt(state, actions, action)
                    action_prompts.append(prompt)
                    metadata.append((i, action.strip()))

            # --- TRACE POINT 1: Prompts ---
            if self.config.debug_mode:
                print(f"[TRACE-REF] Prompts Hash: {hash(tuple(action_prompts))}")
            
            # Tokenize all prompts
            inputs = self.tokenize_prompts(action_prompts)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # --- TRACE POINT 2: Tokenized Input Tensor ---
            if self.config.debug_mode:
                print(f"[TRACE-REF] Input Tensor Sum: {inputs['input_ids'].sum().item()}")
            
            # Get reference model logits (FP16 if configured)
            # with autocast(self.model.device.type, enabled=self.config.reference_fp16):
            ref_outputs = self.reference_model(**inputs)
            ref_logits = ref_outputs.logits

            # --- TRACE POINT 3: Raw Logits Tensor ---
            if self.config.debug_mode:
                print(f"[TRACE-REF] Raw Logits Sum: {ref_logits.sum().item()}")

            reference_logprobs = self.compute_action_scores(
                ref_logits,
                metadata,
                len(states),
                temperature,
            )
            
            return reference_logprobs
    
    def compute_kl_divergence(
        self, 
        current_action_logprobs: List[torch.Tensor],
        reference_action_logprobs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence between current and reference policies"""
        kl_divs = []
        
        for curr_logprobs, ref_logprobs in zip(current_action_logprobs, reference_action_logprobs):            
            # Compute KL(current || reference)
            # KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
            kl = (torch.exp(curr_logprobs) * (curr_logprobs - ref_logprobs)).sum()
            kl_divs.append(kl)
        
        return torch.stack(kl_divs).mean()
