# tests/integration/test_training_sensitive_differences.py

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import hashlib
import pickle

from trainer.experience_roller import ExperienceRoller as OriginalRoller
from trainer.refactored_experience_roller import ExperienceRoller as RefactoredRoller
from tests.mocks.mock_env import MockTextWorldEnvironment
from tests.mocks.mock_policy import MockPolicyBuilder
from tests.mocks.mock_tokenizer import MockTokenizerFactory
from tests.mocks.mock_config import MockPPOConfig


class TestTrainingSensitiveDifferences:
    """Tests specifically designed to catch differences that affect model training."""

    @pytest.fixture
    def realistic_setup(self):
        """Setup with more realistic, non-deterministic components."""
        config = MockPPOConfig()
        config.num_envs = 4
        config.num_steps = 10
        config.max_length = 200
        
        # Use a policy that has some realistic numerical behavior
        policy = MockPolicyBuilder.create_realistic_policy()
        tokenizer = MockTokenizerFactory.create_realistic_tokenizer()
        
        return config, policy, tokenizer

    def test_tensor_operation_sequence_consistency(self, realistic_setup):
        """Test that both implementations perform tensor operations in the same sequence."""
        config, policy, tokenizer = realistic_setup
        
        # Track all tensor operations
        original_operations = []
        refactored_operations = []
        
        def track_tensor_ops(op_list):
            def tensor_op_wrapper(original_func):
                def wrapper(*args, **kwargs):
                    result = original_func(*args, **kwargs)
                    if hasattr(result, 'shape'):  # Is a tensor
                        op_list.append({
                            'func': original_func.__name__,
                            'shape': tuple(result.shape),
                            'dtype': str(result.dtype),
                            'first_few_values': result.flatten()[:5].tolist() if result.numel() > 0 else []
                        })
                    return result
                return wrapper
            return tensor_op_wrapper
        
        # Patch key tensor operations
        tensor_ops_to_track = ['softmax', 'log_softmax', 'multinomial', 'argmax', 'gather', 'cat']
        
        # Test original
        torch.manual_seed(42)
        np.random.seed(42)
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
            original_patches = {}
            for op in tensor_ops_to_track:
                if hasattr(torch, op):
                    original_patches[op] = patch.object(torch, op, track_tensor_ops(original_operations)(getattr(torch, op)))
            
            for patcher in original_patches.values():
                patcher.start()
            
            try:
                original = OriginalRoller(policy, config, torch.device("cpu"), tokenizer)
                orig_buffer, _ = original.run(
                    envs=original.train_envs,
                    num_episodes=2,
                    temperature=0.1,
                    is_eval_mode=False
                )
            finally:
                for patcher in original_patches.values():
                    patcher.stop()
        
        # Test refactored with same seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        with patch('trainer.refactored_experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
            refactored_patches = {}
            for op in tensor_ops_to_track:
                if hasattr(torch, op):
                    refactored_patches[op] = patch.object(torch, op, track_tensor_ops(refactored_operations)(getattr(torch, op)))
            
            for patcher in refactored_patches.values():
                patcher.start()
            
            try:
                refactored = RefactoredRoller(policy, config, torch.device("cpu"), tokenizer)
                ref_buffer, _ = refactored.run(is_eval_mode=False, temperature=0.1)
            finally:
                for patcher in refactored_patches.values():
                    patcher.stop()
        
        # Compare tensor operation sequences
        assert len(original_operations) == len(refactored_operations), \
            f"Different number of tensor operations: {len(original_operations)} vs {len(refactored_operations)}"
        
        for i, (orig_op, ref_op) in enumerate(zip(original_operations[:20], refactored_operations[:20])):
            assert orig_op['func'] == ref_op['func'], f"Operation {i}: different function {orig_op['func']} != {ref_op['func']}"
            assert orig_op['shape'] == ref_op['shape'], f"Operation {i}: different tensor shape"
            
            # Check numerical values (allowing for small floating point differences)
            for j, (orig_val, ref_val) in enumerate(zip(orig_op['first_few_values'], ref_op['first_few_values'])):
                assert abs(orig_val - ref_val) < 1e-6, f"Operation {i}, value {j}: significant numerical difference"

    def test_buffer_serialization_consistency(self, realistic_setup):
        """Test that buffer contents are identical when serialized."""
        config, policy, tokenizer = realistic_setup
        
        def get_buffer_hash(buffer):
            # Create a deterministic hash of buffer contents
            serializable_buffer = []
            for entry in buffer:
                serializable_entry = {
                    'env_idx': entry['env_idx'],
                    'state': entry['state'],
                    'action': entry['action'],
                    'old_logprob': round(entry['old_logprob'], 10),  # Round to avoid floating point issues
                    'value': round(entry['value'], 10),
                    'reward': round(entry['reward'], 10),
                    'finished': entry['finished'],
                    'truncated': entry['truncated'],
                    'valid_token_ids': entry['valid_token_ids']
                }
                serializable_buffer.append(serializable_entry)
            
            return hashlib.md5(pickle.dumps(serializable_buffer, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
        
        # Test multiple seeds to catch stochastic differences
        for seed in [42, 123, 456, 789]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
                original = OriginalRoller(policy, config, torch.device("cpu"), tokenizer)
                orig_buffer, _ = original.run(
                    envs=original.train_envs,
                    num_episodes=2,
                    temperature=0.05,  # Small non-zero temperature
                    is_eval_mode=False
                )
            
            orig_hash = get_buffer_hash(orig_buffer)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            with patch('trainer.refactored_experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
                refactored = RefactoredRoller(policy, config, torch.device("cpu"), tokenizer)
                ref_buffer, _ = refactored.run(is_eval_mode=False, temperature=0.05)
            
            ref_hash = get_buffer_hash(ref_buffer)
            
            assert orig_hash == ref_hash, f"Buffer contents differ for seed {seed}"

    def test_random_state_consumption_pattern(self, realistic_setup):
        """Test that both implementations consume random state in the same pattern."""
        config, policy, tokenizer = realistic_setup
        
        def capture_random_state():
            return {
                'torch_state': torch.get_rng_state(),
                'numpy_state': np.random.get_state(),
                'python_state': hash(str(np.random.random()))  # Quick check
            }
        
        # Track random state at multiple points during execution
        original_states = []
        refactored_states = []
        
        # Mock the policy forward method to capture states
        original_forward = policy.forward
        def track_forward_original(*args, **kwargs):
            original_states.append(capture_random_state())
            return original_forward(*args, **kwargs)
        
        def track_forward_refactored(*args, **kwargs):
            refactored_states.append(capture_random_state())
            return original_forward(*args, **kwargs)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch.object(policy, 'forward', track_forward_original):
            
            original = OriginalRoller(policy, config, torch.device("cpu"), tokenizer)
            original.run(envs=original.train_envs, num_episodes=2, temperature=0.1, is_eval_mode=False)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        with patch('trainer.refactored_experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch.object(policy, 'forward', track_forward_refactored):
            
            refactored = RefactoredRoller(policy, config, torch.device("cpu"), tokenizer)
            refactored.run(is_eval_mode=False, temperature=0.1)
        
        # Compare random state progression
        assert len(original_states) == len(refactored_states), \
            f"Different number of forward calls: {len(original_states)} vs {len(refactored_states)}"
        
        # Check that random states progress identically (at least for first few calls)
        for i in range(min(5, len(original_states))):
            orig_hash = hashlib.md5(original_states[i]['torch_state']).hexdigest()
            ref_hash = hashlib.md5(refactored_states[i]['torch_state']).hexdigest()
            assert orig_hash == ref_hash, f"Random state diverged at call {i}"

    def test_autocast_context_consistency(self, realistic_setup):
        """Test that autocast contexts are used consistently."""
        config, policy, tokenizer = realistic_setup
        
        autocast_entries = []
        autocast_exits = []
        
        # Track autocast usage
        original_autocast = torch.amp.autocast
        
        def track_autocast(*args, **kwargs):
            context = original_autocast(*args, **kwargs)
            
            class TrackedAutocast:
                def __enter__(self):
                    autocast_entries.append(('enter', args, kwargs))
                    return context.__enter__()
                
                def __exit__(self, *exc_info):
                    autocast_exits.append(('exit', args, kwargs))
                    return context.__exit__(*exc_info)
            
            return TrackedAutocast()
        
        torch.manual_seed(42)
        
        with patch('torch.amp.autocast', track_autocast), \
             patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
            
            original = OriginalRoller(policy, config, torch.device("cpu"), tokenizer)
            original.run(envs=original.train_envs, num_episodes=1, temperature=0.1, is_eval_mode=False)
        
        original_entries = autocast_entries.copy()
        original_exits = autocast_exits.copy()
        autocast_entries.clear()
        autocast_exits.clear()
        
        torch.manual_seed(42)
        
        with patch('torch.amp.autocast', track_autocast), \
             patch('trainer.refactored_experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
            
            refactored = RefactoredRoller(policy, config, torch.device("cpu"), tokenizer)
            refactored.run(is_eval_mode=False, temperature=0.1)
        
        # Compare autocast usage patterns
        assert len(original_entries) == len(autocast_entries), "Different autocast entry count"
        assert len(original_exits) == len(autocast_exits), "Different autocast exit count"
        
        for i, (orig_entry, ref_entry) in enumerate(zip(original_entries, autocast_entries)):
            assert orig_entry == ref_entry, f"Autocast entry {i} differs: {orig_entry} != {ref_entry}"

    def test_memory_allocation_patterns(self, realistic_setup):
        """Test that memory allocation patterns are consistent."""
        config, policy, tokenizer = realistic_setup
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory allocation testing")
        
        def get_memory_snapshot():
            if torch.cuda.is_available():
                return {
                    'allocated': torch.cuda.memory_allocated(),
                    'reserved': torch.cuda.memory_reserved()
                }
            return {'allocated': 0, 'reserved': 0}
        
        # Test on GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(device)
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
            original = OriginalRoller(policy, config, device, tokenizer)
            initial_memory = get_memory_snapshot()
            original.run(envs=original.train_envs, num_episodes=2, temperature=0.1, is_eval_mode=False)
            original_final_memory = get_memory_snapshot()
        
        original_memory_diff = {
            k: original_final_memory[k] - initial_memory[k] 
            for k in initial_memory.keys()
        }
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        
        with patch('trainer.refactored_experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
            refactored = RefactoredRoller(policy, config, device, tokenizer)
            initial_memory = get_memory_snapshot()
            refactored.run(is_eval_mode=False, temperature=0.1)
            refactored_final_memory = get_memory_snapshot()
        
        refactored_memory_diff = {
            k: refactored_final_memory[k] - initial_memory[k] 
            for k in initial_memory.keys()
        }
        
        # Allow for some difference in memory usage, but it shouldn't be drastically different
        for key in original_memory_diff.keys():
            diff_ratio = abs(original_memory_diff[key] - refactored_memory_diff[key]) / max(1, abs(original_memory_diff[key]))
            assert diff_ratio < 0.1, f"Memory usage pattern differs significantly for {key}: {original_memory_diff[key]} vs {refactored_memory_diff[key]}"

    def test_step_level_value_consistency(self, realistic_setup):
        """Test that individual step values remain consistent across implementations."""
        config, policy, tokenizer = realistic_setup
        config.num_envs = 1  # Single env for detailed comparison
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
            original = OriginalRoller(policy, config, torch.device("cpu"), tokenizer)
            orig_buffer, _ = original.run(
                envs=original.train_envs,
                num_episodes=1,
                temperature=0.01,  # Very low temperature for more deterministic behavior
                is_eval_mode=False
            )
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        with patch('trainer.refactored_experience_roller.TextWorldEnvironment', MockTextWorldEnvironment):
            refactored = RefactoredRoller(policy, config, torch.device("cpu"), tokenizer)
            ref_buffer, _ = refactored.run(is_eval_mode=False, temperature=0.01)
        
        assert len(orig_buffer) == len(ref_buffer), "Buffer lengths differ"
        
        # Check each individual step for exact consistency
        for i, (orig_entry, ref_entry) in enumerate(zip(orig_buffer, ref_buffer)):
            # These should be exactly equal for deterministic behavior
            assert orig_entry['env_idx'] == ref_entry['env_idx'], f"Step {i}: env_idx differs"
            assert orig_entry['action'] == ref_entry['action'], f"Step {i}: action differs"
            assert orig_entry['state'] == ref_entry['state'], f"Step {i}: state differs"
            assert orig_entry['valid_token_ids'] == ref_entry['valid_token_ids'], f"Step {i}: valid_token_ids differ"
            
            # These should be very close for low temperature
            assert abs(orig_entry['old_logprob'] - ref_entry['old_logprob']) < 1e-5, \
                f"Step {i}: logprob differs significantly: {orig_entry['old_logprob']} vs {ref_entry['old_logprob']}"
            assert abs(orig_entry['value'] - ref_entry['value']) < 1e-5, \
                f"Step {i}: value differs significantly: {orig_entry['value']} vs {ref_entry['value']}"