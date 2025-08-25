# tests/mocks/mock_env.py

from typing import List, Tuple, Dict, Any, Optional
from collections import deque


class MockTextWorldEnvironment:
    """
    Simple mock of TextWorldEnvironment for testing.
    Provides controllable, predictable game behavior without TextWorld dependencies.
    """
    
    def __init__(
        self, 
        config: Optional[Any] = None,
        game_file: str = None,
        is_eval_env: bool = False,
        game_id: int = 0
    ):
        self.config = config
        self.is_eval_env = is_eval_env
        self.game_id = game_id
        
        # Game state tracking
        self.current_step = 0
        self.max_steps = config.num_steps if config else 10
        self.done = False
        self.last_score = 0
        self.step_penalty = config.step_penalty if config else 0.1
        self.history_len = config.history_len if config else 3
        self.history = deque(maxlen=self.history_len)
        
        # Mock game configuration
        self.difficulty = config.difficulty if config else "medium"
        self.repeatable = config.repeatable if config else False
        
        # Controllable test behavior
        self.fixed_actions = ["go north", "take key", "open door", "examine room"]
        self.win_at_step = 5  # Win after 5 steps by default
        self.reward_per_step = 1.0  # Default reward for actions
        
        # Track method calls for testing
        self.reset_count = 0
        self.step_count = 0
        self.close_count = 0
        
        # Configurable behavior
        self.should_win_on_action = None  # Set to specific action to trigger win
        self.custom_state_sequence = None  # Can set sequence of states to return
        self.state_index = 0
        
    def reset(self) -> str:
        """Reset the environment and return initial state."""
        self.reset_count += 1
        self.current_step = 0
        self.done = False
        self.last_score = 0
        self.history.clear()
        self.state_index = 0
        
        # Return initial state
        return self._generate_state()
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action and return (state, reward, done, info)."""
        self.step_count += 1
        self.current_step += 1
        
        # Add to history
        current_state = self._generate_state()
        self.history.append((current_state, action))
        
        # Calculate reward (simple mock logic)
        score_increase = 0
        if "north" in action.lower():
            score_increase = 2
        elif "key" in action.lower():
            score_increase = 3
        elif "door" in action.lower():
            score_increase = 5
        else:
            score_increase = 1
            
        new_score = self.last_score + score_increase
        reward = score_increase - self.step_penalty
        
        # Check win conditions
        won = False
        if self.should_win_on_action and action == self.should_win_on_action:
            self.done = True
            won = True
        elif self.current_step >= self.win_at_step:
            self.done = True
            won = True
        elif self.current_step >= self.max_steps:
            self.done = True
            won = False
        
        self.last_score = new_score
        
        # Generate next state
        self.state_index += 1
        next_state = self._generate_state()
        
        info = {
            "score": self.last_score,
            "admissible_commands": self.get_valid_actions(),
            "won": won,
            "lost": False,
            "steps_taken": self.current_step,
        }
        
        return next_state, reward, self.done, info
    
    def get_valid_actions(self) -> List[str]:
        """Return list of valid actions for current state."""
        if self.done:
            return []
        
        # Return different actions based on step to simulate progression
        if self.current_step == 0:
            return ["go north", "go south", "examine room"]
        elif self.current_step < 3:
            return ["take key", "go north", "go east", "examine door"]
        else:
            return ["open door", "use key", "go north", "examine room"]
    
    def _generate_state(self) -> str:
        """Generate a mock state string similar to real TextWorld format."""
        # Use custom sequence if provided
        if self.custom_state_sequence and self.state_index < len(self.custom_state_sequence):
            return self.custom_state_sequence[self.state_index]
        
        # Build history string
        history_str = "None"
        if self.history:
            history_parts = []
            for state, action in self.history:
                # Simplify state for history
                simple_state = f"Room {self.current_step}"
                history_parts.append(f"{simple_state} > {action}")
            history_str = " | ".join(history_parts[-2:])  # Last 2 actions
        
        # Generate state based on current step
        room_num = (self.current_step % 3) + 1
        current_state = f"Room {room_num}. OBJECTS: key, door. EXITS: north, south"
        
        # Format like real environment
        state = (
            f"Find the treasure and escape.\n"
            f"Before: {history_str}\n"
            f"Now: {current_state}\n"
            f"You are carrying: nothing"
        )
        
        return state
    
    def get_game_stats(self) -> Dict[str, Any]:
        """Get current game statistics."""
        return {
            "difficulty": self.difficulty,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "score": self.last_score,
            "done": self.done,
            "seed": 42  # Fixed seed for mock
        }
    
    def close(self):
        """Close the environment."""
        self.close_count += 1
        self.done = True
    
    # Helper methods for testing
    
    def set_win_condition(self, action: str = None, step: int = None):
        """Configure when the game should end with a win."""
        if action:
            self.should_win_on_action = action
        if step is not None:
            self.win_at_step = step
    
    def set_custom_states(self, states: List[str]):
        """Set a custom sequence of states to return."""
        self.custom_state_sequence = states
        self.state_index = 0
    
    def set_fixed_actions(self, actions: List[str]):
        """Set the valid actions that will be returned."""
        self.fixed_actions = actions


class MockEnvironmentFactory:
    """Factory for creating pre-configured mock environments."""
    
    @staticmethod
    def create_simple_env(config: Any = None) -> MockTextWorldEnvironment:
        """Create a basic environment that wins after 3 steps."""
        env = MockTextWorldEnvironment(config=config)
        env.win_at_step = 3
        return env
    
    @staticmethod
    def create_long_env(config: Any = None) -> MockTextWorldEnvironment:
        """Create an environment that takes many steps."""
        env = MockTextWorldEnvironment(config=config)
        env.win_at_step = 20
        env.max_steps = 25
        return env
    
    @staticmethod
    def create_instant_win_env(config: Any = None) -> MockTextWorldEnvironment:
        """Create an environment that wins on first 'go north' action."""
        env = MockTextWorldEnvironment(config=config)
        env.set_win_condition(action="go north")
        return env
    
    @staticmethod
    def create_deterministic_env(config: Any = None, states: List[str] = None) -> MockTextWorldEnvironment:
        """Create an environment with predetermined state sequence."""
        env = MockTextWorldEnvironment(config=config)
        if states:
            env.set_custom_states(states)
        return env