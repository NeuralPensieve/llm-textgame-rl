# tests/mocks/mock_policy.py

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

class MockPolicy(nn.Module):
    """
    Mock LLMPolicy for testing ExperienceRoller and other components.
    Provides controllable, predictable behavior without loading actual models.
    """
    
    def __init__(
        self, 
        vocab_size: int = 50257,
        hidden_size: int = 768,
        device: str = "cpu",
        deterministic: bool = True,
        use_kl_penalty: bool = False,
        lora_enabled: bool = False,
        disable_value_function: bool = False
    ):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.deterministic = deterministic
        self.use_kl_penalty = use_kl_penalty
        self.lora_enabled = lora_enabled
        self.disable_value_function = disable_value_function
        
        # Mock the model attribute with a config
        self.model = self._create_mock_model()
        
        # Mock value head
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)
        
        # Create reference model if needed
        if use_kl_penalty:
            self.reference_model = self._create_mock_model()
        
        # Track calls for testing
        self.forward_call_count = 0
        self.evaluate_tokens_call_count = 0
        
        # Configurable outputs for testing different scenarios
        self.next_logits = None  # Can be set to return specific logits
        self.next_values = None  # Can be set to return specific values
        self.should_raise_oom = False  # For testing OOM handling
        
    def _create_mock_model(self):
        """Create a mock model with the necessary attributes"""
        class MockModel:
            def __init__(self, vocab_size, hidden_size):
                self.config = type('Config', (), {
                    'vocab_size': vocab_size,
                    'hidden_size': hidden_size
                })()
                
            def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
                batch_size, seq_len = input_ids.shape
                
                # Mock outputs
                logits = torch.randn(batch_size, seq_len, self.config.vocab_size)
                
                hidden_states = None
                if output_hidden_states:
                    # Create fake hidden states
                    hidden_states = [torch.randn(batch_size, seq_len, self.config.hidden_size)]
                
                return type('Output', (), {
                    'logits': logits,
                    'hidden_states': hidden_states
                })()
                
            def parameters(self):
                # Return a dummy parameter for optimizer tests
                return [torch.nn.Parameter(torch.randn(10, 10))]
                
            def print_trainable_parameters(self):
                print("Mock: 100 trainable parameters")
                
        return MockModel(self.vocab_size, self.hidden_size)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mock forward pass that returns controllable outputs.
        """
        self.forward_call_count += 1
        
        # Simulate CUDA OOM if configured
        if self.should_raise_oom:
            self.should_raise_oom = False  # Reset after raising once
            raise torch.cuda.OutOfMemoryError("Mock CUDA OOM")
        
        batch_size, seq_len = input_ids.shape
        
        # Return configured logits or generate random ones
        if self.next_logits is not None:
            logits = self.next_logits
            self.next_logits = None  # Reset after use
        else:
            if self.deterministic:
                # Generate deterministic logits based on input shape
                torch.manual_seed(batch_size * seq_len)
            logits = torch.randn(batch_size, seq_len, self.vocab_size, device=self.device)
        
        # Return configured values or generate random ones
        if self.next_values is not None:
            values = self.next_values
            self.next_values = None  # Reset after use
        elif self.disable_value_function:
            values = torch.zeros((batch_size, 1), device=self.device)
        else:
            if self.deterministic:
                torch.manual_seed(batch_size)
            values = torch.randn(batch_size, 1, device=self.device)
        
        return logits, values
    
    def evaluate_tokens(
        self,
        composite_states: List[Tuple[str, str]],
        chosen_tokens: List[int],
        valid_token_ids: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Mock evaluation of tokens for PPO updates.
        """
        self.evaluate_tokens_call_count += 1
        
        batch_size = len(composite_states)
        
        # Generate mock outputs
        if self.deterministic:
            torch.manual_seed(42)
            
        chosen_log_probs = torch.randn(batch_size, device=self.device)
        values = torch.randn(batch_size, device=self.device)
        
        # Calculate mock entropy based on number of valid tokens
        entropies = []
        for valid_ids in valid_token_ids:
            # Higher entropy for more valid tokens
            entropy = np.log(len(valid_ids)) * 0.5
            entropies.append(entropy)
        entropy = torch.tensor(entropies, device=self.device)
        
        # Mock sequence length
        batch_seq_len = 50  
        
        return chosen_log_probs, values, entropy, batch_seq_len
    
    def get_reference_token_logprobs(
        self,
        composite_states: List[Tuple[str, str]],
        chosen_tokens: List[int]
    ) -> torch.Tensor:
        """
        Mock reference model log probabilities for KL divergence.
        """
        if not self.use_kl_penalty:
            raise ValueError("KL penalty is not enabled in mock")
            
        batch_size = len(composite_states)
        
        if self.deterministic:
            torch.manual_seed(24)
            
        return torch.randn(batch_size, device=self.device)
    
    def compute_kl_divergence(
        self,
        current_logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """
        Mock KL divergence computation.
        """
        # Simple mock: just return mean difference
        return torch.abs(current_logprobs - reference_logprobs).mean()
    
    def get_separate_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Mock parameter groups for optimizer.
        """
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        value_params = list(self.value_head.parameters())
        
        if self.lora_enabled:
            return [
                {"params": mock_params, "lr": 1e-4, "name": "lora_adapters"},
                {"params": value_params, "lr": 1e-3, "name": "value_head"},
            ]
        else:
            return [
                {"params": mock_params, "lr": 1e-4, "name": "pretrained"},
                {"params": value_params, "lr": 1e-3, "name": "value_head"},
            ]
    
    def to(self, device):
        """Mock device movement"""
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self
    
    # Helper methods for testing
    
    def set_next_outputs(self, logits: Optional[torch.Tensor] = None, values: Optional[torch.Tensor] = None):
        """Configure the next forward pass outputs for testing specific scenarios."""
        self.next_logits = logits
        self.next_values = values
    
    def set_deterministic_seed(self, seed: int):
        """Set a specific seed for deterministic outputs."""
        torch.manual_seed(seed)
    
    def enable_oom_simulation(self):
        """Enable CUDA OOM simulation for the next forward pass."""
        self.should_raise_oom = True
    
    def reset_call_counts(self):
        """Reset call counters for testing."""
        self.forward_call_count = 0
        self.evaluate_tokens_call_count = 0


class MockPolicyBuilder:
    """
    Builder pattern for creating MockPolicy instances with specific configurations.
    Useful for creating different test scenarios.
    """
    
    @staticmethod
    def create_simple_policy(device: str = "cpu") -> MockPolicy:
        """Create a basic mock policy for simple tests."""
        return MockPolicy(
            vocab_size=100,  # Small vocab for faster tests
            hidden_size=64,
            device=device,
            deterministic=True
        )
    
    @staticmethod
    def create_policy_with_kl(device: str = "cpu") -> MockPolicy:
        """Create a mock policy with KL penalty enabled."""
        return MockPolicy(
            vocab_size=100,
            hidden_size=64,
            device=device,
            deterministic=True,
            use_kl_penalty=True
        )
    
    @staticmethod
    def create_policy_with_lora(device: str = "cpu") -> MockPolicy:
        """Create a mock policy with LoRA enabled."""
        return MockPolicy(
            vocab_size=100,
            hidden_size=64,
            device=device,
            deterministic=True,
            lora_enabled=True
        )
    
    @staticmethod
    def create_non_deterministic_policy(device: str = "cpu") -> MockPolicy:
        """Create a mock policy with random outputs for testing stochastic behavior."""
        return MockPolicy(
            vocab_size=100,
            hidden_size=64,
            device=device,
            deterministic=False
        )
    
    @staticmethod
    def create_oom_test_policy(device: str = "cpu") -> MockPolicy:
        """Create a mock policy for testing OOM handling."""
        policy = MockPolicy(
            vocab_size=100,
            hidden_size=64,
            device=device,
            deterministic=True
        )
        policy.enable_oom_simulation()
        return policy