import textworld
import re
import random
import os

class TextWorldEnvironment:
    """Wrapper for TextWorld environment"""
    def __init__(self, game_file: str = None, seed: int = None):
        self.max_steps = 50
        self.current_step = 0
        self.done = False
        self.last_score = 0
        self.game_state = None
        self.seed = seed if seed is not None else random.randint(1, 1000000)
        
        if game_file:
            # Initialize with a real TextWorld game file (e.g., zork1.z5)
            request_infos = textworld.EnvInfos(admissible_commands=True, inventory=True)
            self.env = textworld.start(game_file, request_infos=request_infos)
            self.is_mock = False
        else:
            # Create a simple TextWorld game if no game file is provided
            self.env = self._create_simple_game()
            self.is_mock = True
    
    def _create_simple_game(self):
        """Create a simple TextWorld game with a unique seed for testing"""
        options = textworld.GameOptions()
        options.seeds = self.seed
        # options.nb_rooms = random.randint(3, 6)  # Random number of rooms (3 to 6)
        # options.nb_objects = random.randint(3, 6)  # Random number of objects (3 to 6)
        options.nb_rooms = 4
        options.nb_objects = 4
        options.nb_quest_items = 2  # At least one quest item
        options.theme = "house"
        # Use a unique file path based on the seed
        options.path = f"./games/simple_game_{self.seed}.z8"
        
        # Randomly select quest items (1 to 3 items to collect)
        possible_items = ["key", "book", "coin", "lamp", "apple", "knife"]
        num_quest_items = random.randint(1, 3)
        quest_items = random.sample(possible_items, num_quest_items)
        options.quests = [{"goal": "collect", "items": quest_items}]
        
        # Remove existing game file to prevent conflicts
        if os.path.exists(options.path):
            os.remove(options.path)
        
        game_file, _ = textworld.make(options)
        request_infos = textworld.EnvInfos(admissible_commands=True, inventory=True)
        
        # Use TextWorld directly
        env = textworld.start(game_file, request_infos=request_infos)
        return env
    
    def reset(self):
        """Reset environment and return initial state"""
        self.current_step = 0
        self.done = False
        self.last_score = 0
        
        # Regenerate the game with a new random seed if mock environment
        # if self.is_mock:
        #     self.seed = random.randint(1, 1000000)
        #     self.env = self._create_simple_game()
        
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
        reward = reward - 0.01
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            done = True
            
        self.done = done
        self.last_score = self.game_state.score
        
        # Get admissible commands
        admissible_commands = []
        if hasattr(self.game_state, 'admissible_commands'):
            admissible_commands = self.game_state.admissible_commands or []
        elif hasattr(self.game_state, '__getitem__'):
            admissible_commands = self.game_state.get('admissible_commands', [])
        
        return state, reward, done, {
            'score': self.last_score, 
            'admissible_commands': admissible_commands,
            'won': self.game_state.won,
            'lost': self.game_state.lost
        }
    
    def get_valid_actions(self):
        """Get list of valid actions"""
        try:
            if self.game_state:
                if hasattr(self.game_state, 'admissible_commands'):
                    return self.game_state.admissible_commands or []
                if hasattr(self.game_state, '__getitem__'):
                    return self.game_state.get('admissible_commands', [])
        except:
            pass
        
        # Fallback valid actions
        return [
            "look", "inventory", "north", "south", "east", "west",
            "take all", "drop all", "examine room", "wait"
        ]
    
    def _extract_state(self, observation, game_state):
        """Extract state description from game state"""
        if observation is None:
            raise ValueError("Invalid game state: No observation provided.")
        
        description = self._clean_text(observation)

        if game_state:
            inventory = getattr(game_state, 'inventory', None) \
                        or (hasattr(game_state, '__getitem__') and game_state.get('inventory'))
            if inventory:
                description += f" Inventory: {inventory}."
        
        return description
    
    def _clean_text(self, text: str) -> str:
        """Remove ASCII art, banners, and ANSI codes."""
        if not text:
            return text

        # Remove ANSI escape sequences
        text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)

        lines = text.splitlines()
        clean_lines = []

        found_real_text = False
        for line in lines:
            if not found_real_text:
                if len(re.findall(r'[A-Za-z]{2,}', line)) >= 3:
                    found_real_text = True
                    clean_lines.append(line)
            else:
                clean_lines.append(line)

        # Remove trailing inventory prompt clutter
        cleaned = "\n".join(clean_lines)
        cleaned = re.sub(r'>\s*-=\s*\w+\s*=-.*$', '', cleaned, flags=re.MULTILINE)

        return cleaned.strip()
    
    def close(self):
        """Close the environment"""
        if hasattr(self.env, 'close'):
            self.env.close()