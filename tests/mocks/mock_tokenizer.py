# tests/mocks/mock_tokenizer.py

import torch
from typing import List, Dict, Any, Optional


class MockTokenizer:
    """Mock AutoTokenizer for testing without transformers dependency."""
    
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "left"
        # Simple vocab for testing
        self.vocab = {
            "[PAD]": 0,
            "[START]": 1,
            "[EOS]": 2,
            ">": 3,
            "go": 4,
            "north": 5,
            "take": 6,
            "key": 7,
            "open": 8,
            "door": 9,
            "examine": 10,
            "room": 11,
            # Add more as needed
        }
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Simple mock encoding - just returns predictable token IDs."""
        # Very simple tokenization - split by spaces and map to IDs
        tokens = text.lower().split()
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(1)  # [START]
            
        for token in tokens:
            # Use token ID if in vocab, otherwise use a default ID
            token_id = self.vocab.get(token, 12 + (len(token) % 50))
            token_ids.append(token_id)
            
        if add_special_tokens:
            token_ids.append(2)  # [EOS]
            
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Simple mock decoding - reverse of encode."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = []
        
        try:
            for token_id in token_ids:
                # Skip special tokens if requested.
                if skip_special_tokens and token_id in {0, 1, 2, 3}:  # PAD, START, EOS, >
                    continue
                    
                token = inv_vocab.get(token_id, f"[UNK_{token_id}]")
                tokens.append(token)
        except Exception as e:
            print(f"Error decoding token IDs {token_ids}: {e}")
        
        return "".join(tokens) if tokens else ""
    
    def __call__(self, text, **kwargs):
        """Mock the tokenizer call."""
        if isinstance(text, str):
            return {"input_ids": self.encode(text, kwargs.get("add_special_tokens", True))}
        elif isinstance(text, list):
            # Handle batch tokenization
            return self.pad(
                {"input_ids": [self.encode(t, kwargs.get("add_special_tokens", True)) for t in text]},
                **kwargs
            )
    
    def pad(self, encoded_inputs: Dict, padding: str = "longest", return_tensors: str = None, **kwargs):
        """Mock padding functionality."""
        input_ids_list = encoded_inputs.get("input_ids", [])
        
        if not input_ids_list:
            return encoded_inputs
            
        # Find max length for padding
        max_length = max(len(ids) for ids in input_ids_list) if padding == "longest" else 100
        
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids_list:
            if self.padding_side == "left":
                # Pad on the left
                padding_length = max_length - len(ids)
                padded_ids = [self.pad_token_id] * padding_length + ids
                attention_mask = [0] * padding_length + [1] * len(ids)
            else:
                # Pad on the right
                padded_ids = ids + [self.pad_token_id] * (max_length - len(ids))
                attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
                
            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)
        
        result = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_masks
        }
        
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(padded_input_ids)
            result["attention_mask"] = torch.tensor(attention_masks)
            
        return result


class MockTokenizerHelper:
    """Mock TokenizerHelper for testing."""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.tokenizer = MockTokenizer()
        self.tokenizer.padding_side = "left"
        
        # Set max_length from config or use default
        self.max_length = config.max_length if config and hasattr(config, 'max_length') else 50
        
        # Track calls for testing
        self.tokenize_prompts_call_count = 0
        
    def tokenize_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Mock tokenization with simple truncation logic.
        Returns properly formatted tensors like the real tokenizer.
        """
        self.tokenize_prompts_call_count += 1
        
        truncated_prompts = []
        
        for prompt in prompts:
            # Simple mock of truncation - just check length
            # For testing, we'll use character count as a proxy for token count
            if len(prompt) > self.max_length * 5:  # Assume ~5 chars per token
                # Simulate history truncation
                if "Before:" in prompt and "Now:" in prompt:
                    # Find and truncate history section
                    before_idx = prompt.index("Before:")
                    now_idx = prompt.index("Now:")
                    history = prompt[before_idx + 7:now_idx].strip()
                    
                    # Simple truncation - take last half of history
                    if " | " in history:
                        history_parts = history.split(" | ")
                        new_history = " | ".join(history_parts[len(history_parts)//2:])
                    else:
                        new_history = "None"
                    
                    prompt = prompt[:before_idx + 7] + " " + new_history + "\n" + prompt[now_idx:]
            
            truncated_prompts.append(prompt)
        
        # Use the mock tokenizer to create padded output
        result = self.tokenizer(
            truncated_prompts,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return result


def tokenize_actions_for_trie(actions: List[str], tokenizer: MockTokenizer) -> List[List[int]]:
    """
    Mock version of tokenize_actions_for_trie helper function.
    Tokenizes a list of actions for Trie construction.
    """
    tokenized_actions = []
    
    for action in actions:
        # Add ">" prefix and tokenize
        action_with_prefix = f">{action}"
        tokens = tokenizer.encode(action_with_prefix, add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)  # Add EOS token
        tokenized_actions.append(tokens)
    
    return tokenized_actions


class MockTokenizerFactory:
    """Factory for creating different tokenizer configurations."""
    
    @staticmethod
    def create_simple_tokenizer(max_length: int = 50) -> MockTokenizerHelper:
        """Create a basic tokenizer helper with default settings."""
        config = type('Config', (), {'max_length': max_length})()
        return MockTokenizerHelper(config)
    
    @staticmethod
    def create_tokenizer_with_small_context(max_length: int = 10) -> MockTokenizerHelper:
        """Create a tokenizer with very small context for testing truncation."""
        config = type('Config', (), {'max_length': max_length})()
        return MockTokenizerHelper(config)
    
    @staticmethod
    def create_deterministic_tokenizer() -> MockTokenizerHelper:
        """Create a tokenizer that always returns the same token IDs for testing."""
        helper = MockTokenizerHelper()
        # Override encode to return fixed sequences
        original_encode = helper.tokenizer.encode
        
        def deterministic_encode(text, add_special_tokens=True):
            # Return predictable token IDs based on text length
            base_ids = list(range(3, 3 + min(len(text.split()), 10)))
            if add_special_tokens:
                return [1] + base_ids + [2]
            return base_ids
        
        helper.tokenizer.encode = deterministic_encode
        return helper