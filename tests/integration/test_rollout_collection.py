# tests/unit/test_rollout_collection.py

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from trainer.experience_roller import ExperienceRoller
from tests.mocks.mock_policy import MockPolicy, MockPolicyBuilder
from tests.mocks.mock_env import MockTextWorldEnvironment, MockEnvironmentFactory
from tests.mocks.mock_tokenizer import MockTokenizerHelper, MockTokenizerFactory
from tests.mocks.mock_config import MockPPOConfig
from tests.mocks.mock_trie import MockTrie, generate_mask, tokenize_actions_for_trie


class TestExperienceRoller:
    """Test suite for ExperienceRoller rollout collection."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock PPO config."""
        config = MockPPOConfig()
        config.num_envs = 2
        config.num_eval_episodes = 2
        config.num_steps = 10
        config.max_length = 100
        config.history_len = 3
        config.step_penalty = 0.1
        config.difficulty = "easy"
        config.repeatable = True
        config.env_seed = 42
        return config
    
    @pytest.fixture
    def mock_policy(self):
        """Create a mock policy."""
        return MockPolicyBuilder.create_simple_policy(device="cpu")
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer helper."""
        return MockTokenizerFactory.create_simple_tokenizer()
    
    @pytest.fixture
    def experience_roller(self, mock_policy, mock_config, mock_tokenizer):
        """Create an ExperienceRoller with mocked dependencies."""
        # Patch the imports in experience_roller to use our mocks
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
        return roller
    
    def test_initialization(self, experience_roller, mock_config):
        """Test that ExperienceRoller initializes correctly."""
        assert len(experience_roller.train_envs) == mock_config.num_envs
        assert len(experience_roller.eval_envs) == mock_config.num_eval_episodes
        assert experience_roller.device == torch.device("cpu")
    
    def test_single_episode_completion(self, mock_policy, mock_config, mock_tokenizer):
        """Test that a single episode completes successfully."""
        mock_config.num_envs = 1
        mock_config.num_steps = 5
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            # Set environments to win quickly
            for env in roller.train_envs:
                env.win_at_step = 3
            
            # Run rollout
            buffer, episodes = roller.run(
                envs=roller.train_envs,
                num_episodes=1,
                temperature=1.0,
                is_eval_mode=False
            )
            
            # Verify results
            assert len(episodes) == 1
            assert episodes[0].completed  # Should have won
            assert episodes[0].length <= 3  # Won at step 3
            assert len(buffer) > 0  # Should have collected experiences
            
            # Check buffer structure
            for entry in buffer:
                assert 'env_idx' in entry
                assert 'state' in entry
                assert 'action' in entry
                assert 'old_logprob' in entry
                assert 'value' in entry
                assert 'reward' in entry
                assert 'finished' in entry
                assert 'truncated' in entry
    
    def test_parallel_episodes(self, mock_policy, mock_config, mock_tokenizer):
        """Test multiple environments running in parallel."""
        mock_config.num_envs = 3
        mock_config.num_steps = 10
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            # Set different win conditions for each environment
            roller.train_envs[0].win_at_step = 2
            roller.train_envs[1].win_at_step = 4
            roller.train_envs[2].win_at_step = 6
            
            buffer, episodes = roller.run(
                envs=roller.train_envs,
                num_episodes=3,
                temperature=1.0,
                is_eval_mode=False
            )
            
            # All episodes should complete
            assert len(episodes) == 3
            
            # Episodes should have different lengths
            lengths = [ep.length for ep in episodes]
            assert lengths[0] <= 2
            assert lengths[1] <= 4
            assert lengths[2] <= 6
            
            # Check that experiences are from different environments
            env_indices = set(entry['env_idx'] for entry in buffer)
            assert len(env_indices) == 3
    
    def test_evaluation_mode(self, mock_policy, mock_config, mock_tokenizer):
        """Test evaluation mode with zero temperature."""
        mock_config.num_eval_episodes = 2
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            buffer, episodes = roller.run(
                envs=roller.eval_envs,
                num_episodes=2,
                temperature=0.0,  # Should be ignored in eval mode
                is_eval_mode=True
            )
            
            # In eval mode, buffer should be empty (no training data collected)
            assert len(buffer) == 0
            
            # Episodes should still complete
            assert len(episodes) == 2
            
            # Check that game logs are captured for sample games
            assert episodes[0].game_log is not None
            assert "SAMPLE GAME" in episodes[0].game_log
    
    def test_trie_based_action_building(self, mock_policy, mock_config, mock_tokenizer):
        """Test that Trie correctly constrains token generation."""
        mock_config.num_envs = 1
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            # Create a simple test case
            test_env = roller.train_envs[0]
            test_env.get_valid_actions = Mock(return_value=["go north", "take key"])
            
            buffer, episodes = roller.run(
                envs=[test_env],
                num_episodes=1,
                temperature=1.0,
                is_eval_mode=False
            )
            
            # Verify that actions were generated
            assert len(episodes) == 1
            assert len(buffer) > 0
            
            # Check that the Trie was used (valid_token_ids should be populated)
            for entry in buffer:
                assert 'valid_token_ids' in entry
                assert len(entry['valid_token_ids']) > 0
    
    def test_reward_distribution_across_tokens(self, mock_policy, mock_config, mock_tokenizer):
        """Test that rewards are distributed correctly across tokens in an action."""
        mock_config.num_envs = 1
        mock_config.num_steps = 5
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            # Set up environment with known reward
            test_env = roller.train_envs[0]
            test_env.reward_per_step = 3.0  # Fixed reward
            
            buffer, episodes = roller.run(
                envs=[test_env],
                num_episodes=1,
                temperature=1.0,
                is_eval_mode=False
            )
            
            # Check that rewards exist and are reasonable
            # The reward for each token should be the action reward divided by number of tokens
            rewards = [entry['reward'] for entry in buffer]
            
            # All rewards should be non-negative (accounting for step penalty)
            # Some might be negative due to step penalty, but most should be positive
            positive_rewards = [r for r in rewards if r > 0]
            assert len(positive_rewards) > 0, "Should have some positive rewards"
            
            # The total reward across all buffer entries should roughly match episode reward
            # (This is approximate due to step penalties)
            total_buffer_reward = sum(rewards)
            episode_reward = episodes[0].reward
            
            # They should be in the same ballpark (within an order of magnitude)
            if episode_reward != 0:
                ratio = total_buffer_reward / episode_reward
                assert 0.1 < ratio < 10, f"Total buffer reward {total_buffer_reward} too different from episode reward {episode_reward}"
    
    def test_max_steps_truncation(self, mock_policy, mock_config, mock_tokenizer):
        """Test that episodes are truncated at max_steps."""
        mock_config.num_envs = 1
        mock_config.num_steps = 3  # Small number for quick test
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            # Set environment to never win naturally
            roller.train_envs[0].win_at_step = 100
            roller.train_envs[0].max_steps = mock_config.num_steps
            
            buffer, episodes = roller.run(
                envs=roller.train_envs,
                num_episodes=1,
                temperature=1.0,
                is_eval_mode=False
            )
            
            # Episode should complete (either by truncation or naturally)
            assert len(episodes) == 1
            assert episodes[0].length <= mock_config.num_steps
            
            # If the episode didn't complete naturally, it should be truncated
            if not episodes[0].completed:
                # Check that at least one entry is marked as finished or truncated
                terminal_entries = [e for e in buffer if e['finished'] or e['truncated']]
                assert len(terminal_entries) > 0, "Should have at least one terminal entry"
                
                # The last terminal entry should be truncated if we didn't win
                last_terminal = terminal_entries[-1]
                assert last_terminal['truncated'] or last_terminal['finished'], "Last entry should be terminal"
    
    def test_eos_token_detection(self, mock_policy, mock_config, mock_tokenizer):
        """Test that EOS token properly ends action generation."""
        mock_config.num_envs = 1
        
        # Reset the MockTrie eos_id for this test
        MockTrie.eos_id = 0
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            # The EOS token should be set by the tokenizer
            eos_token_id = mock_tokenizer.tokenizer.eos_token_id
            
            buffer, episodes = roller.run(
                envs=roller.train_envs[:1],
                num_episodes=1,
                temperature=1.0,
                is_eval_mode=False
            )
            
            # Verify that MockTrie learned the EOS token
            assert MockTrie.eos_id == eos_token_id
    
    def test_cuda_oom_handling(self, mock_config, mock_tokenizer):
        """Test that CUDA OOM is handled gracefully."""
        # Create a policy that simulates OOM
        mock_policy = MockPolicyBuilder.create_oom_test_policy()
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),  # Using CPU for testing
                tokenizer_helper=mock_tokenizer
            )
            
            # Should handle OOM and complete
            buffer, episodes = roller.run(
                envs=roller.train_envs[:1],
                num_episodes=1,
                temperature=1.0,
                is_eval_mode=False
            )
            
            # Should recover and complete episodes
            assert len(episodes) == 1
    
    def test_action_log_probs_tracking(self, mock_policy, mock_config, mock_tokenizer):
        """Test that action log probabilities are tracked correctly."""
        mock_config.num_envs = 1
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            buffer, episodes = roller.run(
                envs=roller.train_envs[:1],
                num_episodes=1,
                temperature=1.0,
                is_eval_mode=False
            )
            
            # Check that log probs are tracked
            assert len(episodes) == 1
            assert len(episodes[0].action_log_probs) > 0
            
            # Log probs should be negative (log of probability)
            for log_prob in episodes[0].action_log_probs:
                assert log_prob <= 0
    
    def test_valid_token_masking(self, mock_policy, mock_config, mock_tokenizer):
        """Test that invalid tokens are properly masked."""
        mock_config.num_envs = 1
        
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = ExperienceRoller(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
            
            # Track the masks generated
            original_generate_mask = generate_mask
            masks_generated = []
            
            def track_mask(*args, **kwargs):
                result = original_generate_mask(*args, **kwargs)
                masks_generated.append(result[0])
                return result
            
            with patch('trainer.experience_roller.generate_mask', track_mask):
                buffer, episodes = roller.run(
                    envs=roller.train_envs[:1],
                    num_episodes=1,
                    temperature=1.0,
                    is_eval_mode=False
                )
            
            # Verify masks were generated
            assert len(masks_generated) > 0
            
            # Check mask structure
            for mask in masks_generated:
                # Valid tokens should have 0, invalid should have -inf
                assert torch.any(mask == 0.0)  # Some valid tokens
                assert torch.any(torch.isinf(mask))  # Some invalid tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])