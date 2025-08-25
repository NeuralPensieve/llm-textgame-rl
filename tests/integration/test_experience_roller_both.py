# tests/unit/test_experience_roller_both.py

import pytest
import torch
from unittest.mock import patch

from trainer.experience_roller import ExperienceRoller as OriginalRoller
from trainer.refactored_experience_roller import ExperienceRoller as RefactoredRoller
from tests.integration.test_rollout_collection import TestExperienceRoller
from tests.mocks.mock_env import MockTextWorldEnvironment
from tests.mocks.mock_trie import MockTrie, generate_mask, tokenize_actions_for_trie


class TestBothImplementations(TestExperienceRoller):
    """Run all tests against both implementations."""
    
    @pytest.fixture(params=[OriginalRoller, RefactoredRoller], 
                    ids=["original", "refactored"])
    def roller_class(self, request):
        """Provides both ExperienceRoller implementations."""
        return request.param
    
    @pytest.fixture
    def experience_roller(self, roller_class, mock_policy, mock_config, mock_tokenizer):
        """Create an ExperienceRoller using the parameterized class."""
        # Use the same patches for both
        with patch('trainer.experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.refactored_experience_roller.TextWorldEnvironment', MockTextWorldEnvironment), \
             patch('trainer.experience_roller.Trie', MockTrie), \
             patch('trainer.refactored_experience_roller.Trie', MockTrie), \
             patch('trainer.experience_roller.generate_mask', generate_mask), \
             patch('trainer.refactored_experience_roller.generate_mask', generate_mask), \
             patch('trainer.experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie), \
             patch('trainer.refactored_experience_roller.tokenize_actions_for_trie', tokenize_actions_for_trie):
            
            roller = roller_class(
                policy=mock_policy,
                config=mock_config,
                device=torch.device("cpu"),
                tokenizer_helper=mock_tokenizer
            )
        return roller