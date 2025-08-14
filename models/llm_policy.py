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
    ) -> List[torch.Tensor]:
        """Compute action scores based on scoring method"""
        env_action_logprobs = [[] for _ in range(num_states)]

        if self.prompt_manager.scoring_method == "action_token":
            # logprobs = F.log_softmax(logits, dim=-1)

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

                logprobs_i = logits[idx, -n:]
                actions_tensor = torch.tensor(action_tokens, device=logits.device)
                selected = logprobs_i.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                env_action_logprobs[env_idx].append(selected.mean())
        elif self.prompt_manager.scoring_method == "helpful":
            # logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
            helpful_token_id = self.target_tokens["helpful"]
            for (env_idx, action), logit in zip(metadata, logits[:, -1, :]):
                score = logit[helpful_token_id]
                env_action_logprobs[env_idx].append(score)
        else:
            raise ValueError(f"Wrong scoring_method: {self.prompt_manager.scoring_method}")

        env_action_logprobs = [F.log_softmax(torch.stack(row) / self.config.temperature, dim=-1) for row in env_action_logprobs]

        return env_action_logprobs

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

        # Get action scores
        action_inputs = self.tokenize_prompts(action_prompts)
        action_inputs = {k: v.to(self.model.device) for k, v in action_inputs.items()}
        
        with autocast(self.model.device.type):
            action_logits, _ = self.forward(**action_inputs)
        
        env_action_logprobs = self.compute_action_scores(
            action_logits,
            action_inputs["input_ids"],
            metadata,
            len(states),
            action_prompts,
        )

        state_inputs = self.tokenize_prompts(states)  # Just tokenize states directly
        state_inputs = {k: v.to(self.model.device) for k, v in state_inputs.items()}
        
        with autocast(self.model.device.type):
            _, values = self.forward(**state_inputs)  # Use value head output
        
        
        return env_action_logprobs, values.squeeze(-1)

    def evaluate_for_rollout(
        self, states: List[str], action_lists: List[List[str]]
    ) -> Tuple[List[List[float]], List[float]]:
        """Evaluate actions without gradients (for rollout collection)"""
        self.eval()
        env_action_logprobs = []

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
                    # logprobs = F.log_softmax(logits, dim=-1)
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

                        logprobs_i = logits[i, -n:]
                        actions_tensor = torch.tensor(tokens, device=logits.device)
                        selected = logprobs_i.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                        action_scores.append(selected.mean())
                elif self.prompt_manager.scoring_method == "helpful":
                    # logprobs = F.log_softmax(logits[:, -1, :], dim=-1)
                    helpful_token_id = self.target_tokens["helpful"]
                    action_scores = [logit[helpful_token_id] for logit in logits[:, -1, :]]
                else:
                    raise ValueError(f"Wrong scoring_method: {self.prompt_manager.scoring_method}")


                # Apply temperature and normalize (matching compute_action_scores)
                action_scores_tensor = torch.stack(action_scores)
                action_logprobs = F.log_softmax(action_scores_tensor / self.config.temperature, dim=-1)
                env_action_logprobs.append(action_logprobs.cpu().tolist())

        state_inputs = self.tokenize_prompts(states)  # Just states
        state_inputs = {k: v.to(self.model.device) for k, v in state_inputs.items()}
        
        with torch.no_grad(), autocast(self.model.device.type):
            _, values = self.forward(**state_inputs)
            values_list = values.squeeze(-1).cpu().tolist()

        return env_action_logprobs, values_list

    def get_separate_parameter_groups(self):
        """Get parameter groups with different learning rates"""
        return [
            {"params": self.model.parameters(), "lr": self.config.learning_rate, "name": "pretrained"},
            {"params": self.value_head.parameters(), "lr": self.config.learning_rate_value_head, "name": "value_head"},
        ]
