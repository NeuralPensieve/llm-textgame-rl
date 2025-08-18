import textworld
import re
import random
import os
import tempfile
import shutil
from collections import deque

from config import PPOConfig


class TextWorldEnvironment:
    """
    Wrapper for TextWorld environment with conditional logic for repeatable games.
    - In TRAINING (default): If repeatable=True, games are deterministic and reset to their initial state.
    - In EVALUATION (is_eval_env=True): Always creates a new random game on init and reset.
    """
    _current_seed = None # Shared class variable for deterministic training seeds

    def __init__(self, config: PPOConfig, game_file: str = None, is_eval_env: bool = False):
        self.config = config
        self.is_eval_env = is_eval_env
        self.repeatable = config.repeatable
        
        # Initialize the shared seed ONLY ONCE for repeatable training environments
        if self.repeatable and not self.is_eval_env and TextWorldEnvironment._current_seed is None:
            TextWorldEnvironment._current_seed = config.env_seed if config.env_seed is not None else random.randint(1, 1000000)
        self.max_steps = config.num_steps
        self.current_step = 0
        self.done = False
        self.last_score = 0
        self.game_state = None
        self.step_penalty = config.step_penalty
        self.game_file = game_file
        self.temp_dir = None
        self.temp_game_file = None
        self.history_len = config.history_len
        self.history = deque(maxlen=self.history_len)
        self.seed = None # Will be set during game creation

        request_infos = self._get_request_infos()
        if game_file:
            self.env = textworld.start(game_file, request_infos=request_infos)
        else:
            self.env = self._create_simple_game()

    def _get_request_infos(self):
        """Centralized method for requesting game info."""
        return textworld.EnvInfos(
            admissible_commands=True, 
            inventory=True,
            description=True,
            score=True,
            won=True,
            lost=True,
            objective=True,
            location=True,
            entities=True,
        )

    def _cleanup_temp_files(self):
        """Clean up any temporary files and directories"""
        if self.temp_game_file and os.path.exists(self.temp_game_file):
            try:
                os.remove(self.temp_game_file)
            except:
                pass
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
        
        self.temp_game_file, self.temp_dir = None, None

    def _create_simple_game(self):
        """Create a TextWorld game with configurable difficulty in memory"""
        self._cleanup_temp_files()

        # Determine the seed based on the environment's purpose
        if self.is_eval_env or not self.repeatable:
            self.seed = random.randint(1, 1000000)
        else: # This is a repeatable training environment
            TextWorldEnvironment._current_seed += 1
            self.seed = TextWorldEnvironment._current_seed
        options = textworld.GameOptions()
        options.seeds = self.seed
        
        if self.config.difficulty == "easy":
            options.nb_rooms, options.nb_objects = 2, 4
        elif self.config.difficulty == "medium":
            options.nb_rooms, options.nb_objects, options.quest_length = 3, 4, 3
        elif self.config.difficulty == "hard":
            options.nb_rooms, options.nb_objects, options.quest_length = 6, 6, 5
        else:
            options.nb_rooms, options.nb_objects, options.quest_length = 3, 4, 3
            
        options.theme = "house"
        
        # Create temporary directory for this game
        self.temp_dir = tempfile.mkdtemp(prefix="textworld_")
        self.temp_game_file = os.path.join(self.temp_dir, f"game_{self.seed}.z8")
        options.path = self.temp_game_file
        
        # Create the game
        game_file, _ = textworld.make(options)
            
        return textworld.start(game_file, request_infos=self._get_request_infos())

    def reset(self):
        """
        Reset environment. If repeatable, reset the same game. 
        Otherwise, create a new game instance. Returns the initial state dictionary.
        """
        self.current_step, self.done, self.last_score = 0, False, 0
        self.history.clear()

        # Hard reset for evaluation envs or non-repeatable training envs
        if self.is_eval_env or not self.repeatable:
            if hasattr(self, 'env') and self.env is not None:
                self.env.close()
            self._cleanup_temp_files()
            self.env = self._create_simple_game()
        
        # For both hard and soft resets, we need to call the underlying env's reset
        self.game_state = self.env.reset()
        return self._extract_state(self.game_state)

    def step(self, action: str):
        """Take action and return (state_dict, reward, done, info)"""
        self.current_step += 1

        # This will be used to populate the history
        if self.game_state:
            self.history.append((self.game_state.description, action))

        # Send command to the game and get the new game state
        self.game_state, score, done = self.env.step(action)
        reward = (score - self.last_score) - self.step_penalty

        state = self._extract_state(self.game_state)

        self.done = done
        self.last_score = self.game_state.score

        return (
            state,
            reward,
            done,
            {
                "score": self.last_score,
                "admissible_commands": self.game_state.get("admissible_commands", []),
                "won": self.game_state.won,
                "lost": self.game_state.lost,
                "steps_taken": self.current_step,
            },
        )

    def get_valid_actions(self):
        """Get list of valid actions from the current game state"""
        if self.game_state:
            return self.game_state.get("admissible_commands", [])
        raise ValueError(f'No valid actions. Game state is {self.game_state}')
        
    def _build_concise_state(self, game_state) -> str:
        """Builds a compact state string from structured game info."""
        if game_state.infos is None:
            return self._clean_text(game_state.description)

        location_obj = game_state.infos.get("location")
        
        # Get the room name from the object, falling back to the description's first line
        if location_obj and hasattr(location_obj, 'name'):
            location_name = location_obj.name
        else:
            location_name = game_state.get("description", "Unknown Location").split('\n')[0]
            location_name = re.sub(r"-=\s*.*?\s*=-", "", location_name).strip()

        objects = game_state.infos.get("entities", [])

        # Get exits directly from the location object's 'exits' attribute.
        exits = []
        if location_obj and hasattr(location_obj, 'exits'):
            exits = list(location_obj.exits.keys())

        objects_str = f"OBJECTS: {', '.join(objects) if objects else 'None'}"
        exits_str = f"EXITS: {', '.join(exits) if exits else 'None'}"
        
        return f"{location_name}. {objects_str}. {exits_str}"

    def _extract_state(self, game_state):
        """Gathers all state information and formats it as text."""
        if not game_state: raise ValueError("Invalid game state provided.")
        state_dict = {
            "objective": self.clean_objective(game_state.get("objective", "No objective specified.")),
            "previous state actions": [item for pair in self.history for item in pair] if self.history else None,
            "current state": self._build_concise_state(game_state),
            "inventory": self._clean_text(game_state.get("inventory", "You are carrying nothing.")),
            "available actions": game_state.get("admissible_commands", [])
        }
        return self._format_state_as_text(state_dict)

    def _format_state_as_text(self, state_dict: dict) -> str:
        """Converts the state dictionary into a compact, delimited string."""
        history_list = state_dict["previous state actions"]
        history_str = "None"
        if history_list:
            history_parts, last_state = [], None
            for i in range(0, len(history_list), 2):
                state = self._clean_text(history_list[i]).replace("\n", " ")
                action = history_list[i+1]
                if state != last_state:
                    if last_state is not None: history_parts.append(" | ")
                    history_parts.append(f"{state} > {action}")
                else:
                    history_parts.append(f" > {action}")
                last_state = state
            history_str = "".join(history_parts)
        actions_str = " | ".join(state_dict["available actions"])
        return (
            f"OBJECTIVE: {state_dict['objective']}\n"
            f"HISTORY: {history_str}\n"
            f"STATE: {state_dict['current state']}\n"
            f"INVENTORY: {state_dict['inventory']}\n"
            f"ACTIONS: {actions_str}"
        )

    def _clean_text(self, text: str) -> str:
        """Remove ASCII art, banners, and ANSI codes. Normalize spaces and newlines."""
        if not text:
            return ""

        text = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)
        lines = text.splitlines()
        clean_lines = []
        found_real_text = False
        for line in lines:
            stripped_line = line.strip()
            if not found_real_text and re.search(r"[a-zA-Z]", stripped_line):
                found_real_text = True
            
            if found_real_text:
                # Remove decorative lines like '-= Living Room =-'
                if not (stripped_line.startswith("-=") and stripped_line.endswith("=-")):
                    clean_lines.append(line)

        cleaned = "\n".join(clean_lines)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)
        return cleaned.strip()
    
    def clean_objective(self, raw_objective: str) -> str:
        """
        Cleans a TextWorld objective by removing the first and last sentences.
        This is a simple and robust method to remove common fluff.
        """
        # Remove the RAW: prefix and clean up whitespace
        text = re.sub(r"^.*?RAW:\s*", "", raw_objective, flags=re.IGNORECASE).strip()

        # Split the text into sentences using periods, question marks, and exclamation points
        sentences = re.split(r'(?<=[.?!])\s+', text)

        # If there are more than 2 sentences, drop the first and the last one
        if len(sentences) > 2:
            core_sentences = sentences[1:-1]
            return " ".join(core_sentences)
        # If there are 1 or 2 sentences, the objective is likely already simple. Return it as is.
        elif len(sentences) == 2:
            # Sometimes there are two sentences, and sentence 1 is fluff
            if "TextWorld" in sentences[0]:
                return sentences[1]
            
        return text

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
        """Close the environment and clean up all resources"""
        if hasattr(self.env, "close"):
            self.env.close()
        self._cleanup_temp_files()
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion"""
        self.close()


"""
## Recommendations for Improvement

Here are three recommendations, from easiest to most complex, to make your agent even more capable.

1. Abstract the History

Instead of just storing the raw text of the last few turns, you could transition to a more summarized or "abstracted" history.

How: After each action, use a simple rule or a separate LLM call to summarize the outcome of the action.

Example: Instead of HISTORY: You are in the kitchen. There is a table. > take apple, the history could become HISTORY: took apple from kitchen.

Benefit: This keeps the history far more concise while retaining the key information, allowing you to store a longer sequence of important events within the same token budget.

2. Build a World Map

This is the most impactful improvement for navigation tasks. Add a new field to your state that represents the agent's discovered map.

How: In your TextWorldEnvironment wrapper, maintain a dictionary that stores visited rooms and the connections between them. Format this dictionary into a string for the state.

Example State Addition:

MAP: You are in the Kitchen. Exits lead to: Pantry (north), Living Room (south). Discovered: Attic, Pantry.
Or a more compact version:

MAP: Kitchen -> [Pantry(N), LivingRoom(S)]; Pantry -> [Kitchen(S)]; Attic -> [LivingRoom(down)]
Benefit: This gives the agent spatial memory. It can learn to navigate efficiently instead of re-discovering paths, which is critical for solving complex quests.

3. Track Key Objects

To solve puzzles involving items left in other rooms, give the agent a memory of where it has seen important objects.

How: In your wrapper, maintain a list of key items (you might need heuristics to identify them, e.g., anything that can be taken) and their last known locations.

Example State Addition:

SEEN_OBJECTS: silver key (in Pantry), locked chest (in Attic)
Benefit: This provides object persistence. The agent can reason that it needs to go back to the pantry to get the key to open the chest in the attic. This is a crucial step towards more sophisticated problem-solving.
"""