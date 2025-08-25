# Test to verify the skip_special_tokens difference using existing mocks

import pytest
import torch
import numpy as np
from unittest.mock import patch

from trainer.experience_roller import ExperienceRoller as OriginalRoller
from trainer.refactored_experience_roller import ExperienceRoller as RefactoredRoller
from tests.mocks.mock_env import MockTextWorldEnvironment
from tests.mocks.mock_trie import MockTrie, generate_mask, tokenize_actions_for_trie
from tests.mocks.mock_policy import MockPolicyBuilder
from tests.mocks.mock_tokenizer import MockTokenizerFactory
from tests.mocks.mock_config import MockPPOConfig


def test_tokenizer_decode_difference_with_special_tokens():
    """Test if the skip_special_tokens difference causes state divergence."""
    
    config = MockPPOConfig()
    config.num_envs = 1  # Single env for clarity
    config.num_steps = 3
    
    policy = MockPolicyBuilder.create_simple_policy()
    tokenizer_helper = MockTokenizerFactory.create_simple_tokenizer()
    
    # First, let's verify the mock tokenizer behaves differently
    test_tokens_with_special = [4, 2, 5]  # "go" + EOS + "north"
    
    default_decode = tokenizer_helper.tokenizer.decode(test_tokens_with_special)  # default skip_special_tokens=True
    explicit_no_skip = tokenizer_helper.tokenizer.decode(test_tokens_with_special, skip_special_tokens=False)
    
    print(f"Tokenizer test:")
    print(f"  Default decode (skip=True): '{default_decode}'")
    print(f"  Explicit no skip (skip=False): '{explicit_no_skip}'")
    
    if default_decode == explicit_no_skip:
        print("WARNING: Mock tokenizer doesn't show expected difference")
        return False
    
    # Create a scenario that will generate special tokens in paths
    class SpecialTokenMockEnv(MockTextWorldEnvironment):
        def get_valid_actions(self):
            # Return actions that will include special tokens when tokenized
            return ["go", "north"]  # These will get EOS tokens appended
    
    # Test both implementations
    torch.manual_seed(42)
    np.random.seed(42)
    MockTrie.eos_id = 0  # Reset
    
    with patch('trainer.experience_roller.TextWorldEnvironment', SpecialTokenMockEnv), \
         patch('trainer.experience_roller.Trie', MockTrie), \
         patch('trainer.experience_roller.generate_mask', generate_mask), \
         patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie), \
         patch('trainer.experience_roller.format_prompt', lambda x: x):
        
        original = OriginalRoller(policy, config, torch.device("cpu"), tokenizer_helper)
        orig_buffer, _ = original.run(
            envs=original.train_envs,
            num_episodes=1,
            temperature=0.1,
            is_eval_mode=False
        )
    
    torch.manual_seed(42)
    np.random.seed(42)
    MockTrie.eos_id = 0  # Reset
    
    with patch('trainer.refactored_experience_roller.TextWorldEnvironment', SpecialTokenMockEnv), \
         patch('trainer.refactored_experience_roller.Trie', MockTrie), \
         patch('trainer.refactored_experience_roller.generate_mask', generate_mask), \
         patch('trainer.refactored_experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie), \
         patch('trainer.refactored_experience_roller.format_prompt', lambda x: x):
        
        refactored = RefactoredRoller(policy, config, torch.device("cpu"), tokenizer_helper)
        ref_buffer, _ = refactored.run(is_eval_mode=False, temperature=0.1)
    
    # Compare the state fields (specifically the partial command strings)
    print(f"\nBuffer comparison:")
    print(f"  Original buffer length: {len(orig_buffer)}")
    print(f"  Refactored buffer length: {len(ref_buffer)}")
    
    if len(orig_buffer) != len(ref_buffer):
        print("  Buffer lengths differ - fundamental difference detected")
        return True
    
    state_differences_found = False
    for i, (orig_entry, ref_entry) in enumerate(zip(orig_buffer, ref_buffer)):
        orig_state = orig_entry['state']
        ref_state = ref_entry['state']
        
        # state is (observation, partial_command_string)
        orig_partial_cmd = orig_state[1]
        ref_partial_cmd = ref_state[1]
        
        print(f"  Entry {i}:")
        print(f"    Original partial_cmd: '{orig_partial_cmd}'")
        print(f"    Refactored partial_cmd: '{ref_partial_cmd}'")
        print(f"    Action: {orig_entry['action']}")
        
        if orig_partial_cmd != ref_partial_cmd:
            print(f"    *** STATE DIFFERENCE DETECTED ***")
            state_differences_found = True
    
    return state_differences_found


def test_decode_behavior_verification():
    """Verify that the decode difference is actually happening."""
    
    tokenizer_helper = MockTokenizerFactory.create_simple_tokenizer()
    
    # Test with tokens that include special tokens
    # According to mock tokenizer: EOS=2, ">"=3, "go"=4, "north"=5
    test_paths = [
        [3, 4, 2],      # ">" + "go" + EOS
        [3, 5, 2],      # ">" + "north" + EOS
        [4, 5, 2],      # "go" + "north" + EOS
    ]
    
    print("Testing decode behavior with special tokens:")
    for i, path in enumerate(test_paths):
        # Original implementation uses default (skip_special_tokens=True)
        original_decode = tokenizer_helper.tokenizer.decode(path)
        
        # Refactored uses explicit skip_special_tokens=False
        refactored_decode = tokenizer_helper.tokenizer.decode(path, skip_special_tokens=False)
        
        print(f"  Path {i} {path}:")
        print(f"    Original (skip=True): '{original_decode}'")
        print(f"    Refactored (skip=False): '{refactored_decode}'")
        print(f"    Different: {original_decode != refactored_decode}")


def test_cumulative_partial_command_building():
    """Test how partial commands build up differently over multiple tokens."""
    
    tokenizer_helper = MockTokenizerFactory.create_simple_tokenizer()
    
    # Simulate building partial commands step by step
    token_sequence = [
        [3],        # ">"
        [4],        # "go" 
        [2],        # EOS
    ]
    
    print("\nTesting cumulative partial command building:")
    
    # Original approach (skip_special_tokens=True by default)
    original_partial = ""
    for step, tokens in enumerate(token_sequence):
        decoded = tokenizer_helper.tokenizer.decode(tokens)  # default skip=True
        original_partial += decoded
        print(f"  Step {step}: tokens={tokens}, decoded='{decoded}', cumulative='{original_partial}'")
    
    print()
    
    # Refactored approach (skip_special_tokens=False)
    refactored_partial = ""
    for step, tokens in enumerate(token_sequence):
        decoded = tokenizer_helper.tokenizer.decode(tokens, skip_special_tokens=False)
        refactored_partial += decoded
        print(f"  Step {step}: tokens={tokens}, decoded='{decoded}', cumulative='{refactored_partial}'")
    
    print(f"\nFinal comparison:")
    print(f"  Original final partial: '{original_partial}'")
    print(f"  Refactored final partial: '{refactored_partial}'")
    print(f"  Different: {original_partial != refactored_partial}")
    
    return original_partial != refactored_partial


if __name__ == "__main__":
    print("1. Testing basic decode behavior...")
    test_decode_behavior_verification()
    
    print("\n2. Testing cumulative command building...")
    cumulative_diff = test_cumulative_partial_command_building()
    
    print("\n3. Testing full implementation with special tokens...")
    impl_diff = test_tokenizer_decode_difference_with_special_tokens()
    
    if cumulative_diff or impl_diff:
        print("\n✓ TOKENIZER DECODE HYPOTHESIS CONFIRMED")
        print("The skip_special_tokens difference is causing state divergence")
    else:
        print("\n✗ Tokenizer decode hypothesis not confirmed")
        print("No state differences detected")