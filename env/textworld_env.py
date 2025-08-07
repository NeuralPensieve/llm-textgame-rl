import textworld
import re
import random
import os


class TextWorldEnvironment:
    """Wrapper for TextWorld environment with enhanced difficulty settings"""

    def __init__(self, game_file: str = None, seed: int = None, step_penalty: float = 0.1, difficulty: str = "easy"):
        self.max_steps = 50
        self.current_step = 0
        self.done = False
        self.last_score = 0
        self.game_state = None
        self.step_penalty = step_penalty
        self.seed = seed if seed is not None else random.randint(1, 1000000)
        self.difficulty = difficulty

        if game_file:
            # Initialize with a real TextWorld game file (e.g., zork1.z5)
            request_infos = textworld.EnvInfos(admissible_commands=True, inventory=True)
            self.env = textworld.start(game_file, request_infos=request_infos)
        else:
            # Create a simple TextWorld game if no game file is provided
            self.env = self._create_simple_game()

    def _create_simple_game(self):
        """Create a TextWorld game with configurable difficulty"""
        options = textworld.GameOptions()
        options.seeds = self.seed
        
        # Configure difficulty settings
        if self.difficulty == "trivial":
            # Original logic - goal attainable with one action
            options.nb_rooms = 2
            options.nb_objects = 4
            # Don't set quest_length to allow single-action solutions
        elif self.difficulty == "easy":
            options.nb_rooms = 3
            options.nb_objects = 4
            options.quest_length = 3  # Minimum steps to complete
        elif self.difficulty == "medium":
            options.nb_rooms = 6
            options.nb_objects = 6
            options.quest_length = 5
        elif self.difficulty == "hard":
            options.nb_rooms = 10
            options.nb_objects = 8
            options.quest_length = 7
        else:  # default to medium
            options.nb_rooms = 6
            options.nb_objects = 6
            options.quest_length = 5
            
        options.theme = "house"
        options.path = f"./games/game_{self.difficulty}_{self.seed}.z8"

        # Remove existing game file to prevent conflicts
        if os.path.exists(options.path):
            os.remove(options.path)

        # Create the game - quest_length is the key parameter for complexity
        game_file, _ = textworld.make(options)
            
        request_infos = textworld.EnvInfos(
            admissible_commands=True, 
            inventory=True,
            description=True,
            score=True,
            won=True,
            lost=True
        )

        env = textworld.start(game_file, request_infos=request_infos)
        return env

    def set_difficulty(self, difficulty: str):
        """Change difficulty and recreate the game"""
        self.difficulty = difficulty
        if hasattr(self.env, 'close'):
            self.env.close()
        self.env = self._create_simple_game()

    def reset(self):
        """Reset environment and return initial state"""
        self.current_step = 0
        self.done = False
        self.last_score = 0

        # Reset the game state
        self.game_state = self.env.reset()
        observation = self.game_state.feedback

        return self._extract_state(observation, self.game_state)

    def step(self, action: str):
        """Take action and return (state, reward, done, info)"""
        if self.done:
            return self.reset(), 0, True, {}

        self.current_step += 1

        # Send command to the game
        self.game_state = self.env.step(action)[0]
        observation = self.game_state.feedback
        reward = self.game_state.score - self.last_score
        done = self.game_state.done

        state = self._extract_state(observation, self.game_state)

        # Add step penalty to encourage efficiency
        reward = reward - self.step_penalty

        # Check if max steps reached
        if self.current_step >= self.max_steps:
            done = True

        self.done = done
        self.last_score = self.game_state.score

        # Get admissible commands
        admissible_commands = []
        if hasattr(self.game_state, "admissible_commands"):
            admissible_commands = self.game_state.admissible_commands or []
        elif hasattr(self.game_state, "__getitem__"):
            admissible_commands = self.game_state.get("admissible_commands", [])

        return (
            state,
            reward,
            done,
            {
                "score": self.last_score,
                "admissible_commands": admissible_commands,
                "won": self.game_state.won,
                "lost": self.game_state.lost,
                "steps_taken": self.current_step,
            },
        )

    def get_valid_actions(self):
        """Get list of valid actions"""
        try:
            if self.game_state:
                if hasattr(self.game_state, "admissible_commands"):
                    return self.game_state.admissible_commands or []
                if hasattr(self.game_state, "__getitem__"):
                    return self.game_state.get("admissible_commands", [])
        except:
            raise ValueError("No admissible commands found")

    def _extract_state(self, observation, game_state):
        """Extract state description from game state"""
        if observation is None:
            raise ValueError("Invalid game state: No observation provided.")

        description = self._clean_text(observation)

        if game_state:
            inventory = getattr(game_state, "inventory", None) or (
                hasattr(game_state, "__getitem__") and game_state.get("inventory")
            )
            if inventory:
                description += f" Inventory: {inventory}."

        return description

    def _clean_text(self, text: str) -> str:
        """Remove ASCII art, banners, and ANSI codes. Normalize spaces and newlines."""
        if not text:
            return text

        # Remove ANSI escape sequences
        text = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)

        lines = text.splitlines()
        clean_lines = []

        found_real_text = False
        for line in lines:
            if not found_real_text:
                if len(re.findall(r"[A-Za-z]{2,}", line)) >= 3:
                    found_real_text = True
                    clean_lines.append(line)
            else:
                clean_lines.append(line)

        # Remove trailing inventory prompt clutter
        cleaned = "\n".join(clean_lines)
        cleaned = re.sub(r">\s*-=\s*\w+\s*=-.*$", "", cleaned, flags=re.MULTILINE)

        # Normalize spaces: replace 3+ consecutive spaces with single space
        cleaned = re.sub(r" {3,}", " ", cleaned)

        # Normalize newlines: replace 2+ consecutive newlines with single newline
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)

        return cleaned.strip()

    def get_game_stats(self):
        """Get current game statistics"""
        return {
            "difficulty": self.difficulty,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "score": self.last_score,
            "done": self.done,
            "seed": self.seed
        }

    def close(self):
        """Close the environment"""
        if hasattr(self.env, "close"):
            self.env.close()