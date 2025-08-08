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

        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model configuration with use_cache=False
        model_config = AutoConfig.from_pretrained(config.model_name, use_cache=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
            device_map="auto",
            max_memory={0: "8GB"},
        )

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

        # Add value head
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        ).to(self.model.device)

        # Always cache target tokens, as 'high' is needed for value scoring
        self._cache_target_tokens()

    def _cache_target_tokens(self):
        """Cache target token IDs for 'helpful' scoring"""
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

    def tokenize_prompts_basic(self, prompts: List[str]):
        """Tokenize prompts efficiently with caching"""
        self.tokenizer.padding_side = "left"

        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_attention_mask=True,
        )

    def tokenize_prompts(self, prompts: List[str]):
        """Tokenize prompts efficiently with middle truncation and dynamic padding."""
        self.tokenizer.padding_side = "left"
        max_len = self.config.max_length
        half_len = max_len // 2

        # 1. Tokenize each prompt individually without padding or truncation
        tokenized_prompts = self.tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]

        # 2. Apply middle truncation only to sequences that are too long
        truncated_ids = []
        for ids in tokenized_prompts:
            if len(ids) > max_len:
                # This sequence is too long, so we truncate it to max_len
                truncated_ids.append(ids[:half_len] + ids[-(max_len - half_len):])
            else:
                truncated_ids.append(ids)

        # 3. Pad all sequences to the length of the LONGEST sequence in this batch
        padded = self.tokenizer.pad(
            {"input_ids": truncated_ids},
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return padded

    def compute_action_scores(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        metadata: List[Tuple],
        num_states: int,
        action_prompts: List[str],
    ) -> List[List[torch.Tensor]]:
        """Compute action scores based on scoring method"""
        env_action_logprobs = [[] for _ in range(num_states)]

        if self.prompt_manager.scoring_method == "action_token":
            logprobs = F.log_softmax(logits, dim=-1)

            for idx, ((env_idx, action), prompt) in enumerate(
                zip(metadata, action_prompts)
            ):
                action_tokens = self.tokenizer.encode(
                    " " + action.strip(), add_special_tokens=False
                )
                n = len(action_tokens)
                prompt_tokens = input_ids[idx][-n:].cpu().tolist()

                if prompt_tokens != action_tokens:
                    raise ValueError(f"Token mismatch for action '{action.strip()}'")

                logprobs_i = logprobs[idx, -n:]
                actions_tensor = torch.tensor(action_tokens, device=logits.device)
                selected = logprobs_i.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                env_action_logprobs[env_idx].append(selected.mean())
        else:
            logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
            helpful_token_id = self.target_tokens["helpful"]
            for (env_idx, action), logprob_dist in zip(metadata, logprobs):
                score = logprob_dist[helpful_token_id]
                env_action_logprobs[env_idx].append(score)

        return env_action_logprobs
    
    def compute_value_scores(self, logprobs: torch.Tensor) -> torch.Tensor:
        """Compute value scores using 'high' token probability"""
        return logprobs[:, self.target_tokens["high"]]

    def evaluate_actions(
        self, states: List[str], action_lists: List[List[str]]
    ) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
        """Evaluate actions with gradients (for training)"""
        self.train()
        
        # Build action prompts
        action_prompts = []
        metadata = []
        for i, (state, actions) in enumerate(zip(states, action_lists)):
            for action in actions:
                prompt = self.prompt_manager.get_action_prompt(state, actions, action)
                action_prompts.append(prompt)
                metadata.append((i, action.strip()))
        
        value_prompts = [self.prompt_manager.get_value_prompt(state)]

        # Get action scores
        action_inputs = self.tokenize_prompts(action_prompts)
        value_inputs = self.tokenize_prompts(value_prompts)
        action_inputs = {k: v.to(self.model.device) for k, v in action_inputs.items()}
        value_inputs = {k: v.to(self.model.device) for k, v in value_inputs.items()}
        
        with autocast(self.model.device.type):
            action_logits, _ = self.forward(**action_inputs)
            value_logits, _ = self.forward(**value_inputs)
            value_logprobs = F.log_softmax(value_logits[:, -1, :], dim=-1)
        
        env_action_logprobs = self.compute_action_scores(
            action_logits,
            action_inputs["input_ids"],
            metadata,
            len(states),
            action_prompts,
        )
        values = self.compute_value_scores(value_logprobs)
        
        return env_action_logprobs, values

    def evaluate_for_rollout(
        self, states: List[str], action_lists: List[List[str]]
    ) -> Tuple[List[List[float]], List[float]]:
        """Evaluate actions without gradients (for rollout collection)"""
        self.eval()
        env_action_logprobs, values_list = [], []

        for state, actions in zip(states, action_lists):
            # Create action prompts
            action_prompts = [
                self.prompt_manager.get_action_prompt(state, actions, action)
                for action in actions
            ]
            
            # Get action scores
            inputs = self.tokenize_prompts(action_prompts)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad(), autocast(self.model.device.type):
                logits, _ = self.forward(**inputs)
                if self.prompt_manager.scoring_method == "action_token":
                    action_scores = []
                    logprobs = F.log_softmax(logits, dim=-1)
                    actions_tokens = [
                        self.tokenizer.encode(f" {a}", add_special_tokens=False)
                        for a in actions
                    ]
                    input_ids = inputs["input_ids"]

                    for i, tokens in enumerate(actions_tokens):
                        n = len(tokens)
                        prompt_tokens = input_ids[i][-n:].cpu().tolist()
                        if prompt_tokens != tokens:
                            raise ValueError(
                                f"Action tokens do not match for action '{actions[i]}'"
                            )

                        logprobs_i = logprobs[i, -n:]
                        actions_tensor = torch.tensor(tokens, device=logprobs.device)
                        selected = logprobs_i.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                        action_scores.append(selected.mean().item())
                else:
                    logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
                    helpful_token_id = self.target_tokens["helpful"]
                    action_scores = [logprob[helpful_token_id].cpu().item() for logprob in logprobs]

            env_action_logprobs.append(action_scores)

            value_prompt = self.prompt_manager.get_value_prompt(state)
            value_inputs = self.tokenize_prompts([value_prompt])
            value_inputs = {k: v.to(self.model.device) for k, v in value_inputs.items()}

            with torch.no_grad(), autocast(self.model.device.type):
                logits, _ = self.forward(**value_inputs)
                logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
                values_list.append(logprobs[0][self.target_tokens["high"]].item())

        return env_action_logprobs, values_list

    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text response (for testing/debugging)"""
        self.eval()

        # Best practice: Use the direct tokenizer with truncation for this simple case.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.model.device)

        with torch.no_grad(), autocast(self.model.device.type):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt) :].strip()

    def get_device(self):
        """Get model device"""
        return self.model.device

    def get_separate_parameter_groups(self):
        """Get parameter groups with different learning rates"""
        return [
            {"params": self.model.parameters(), "lr": 1e-5, "name": "pretrained"},
            {"params": self.value_head.parameters(), "lr": 3e-4, "name": "value_head"},
        ]
