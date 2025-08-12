from typing import List

class PromptManager:
    def __init__(self, scoring_method: str = "helpful"):
        self.scoring_method = scoring_method
        
    def get_action_prompt(self, state: str, actions: List[str], action: str) -> str:
        if self.scoring_method == "helpful":
            # return f"{state}\nConsidering available actions: {', '.join(actions)}\nThis action {action} is helpful"
            return f"{state}\nAction {action} is helpful"
        elif self.scoring_method == "action_token":
            return f"{state}\nBest action is {action}"
        else:
            # Add this to catch any unexpected values
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
        
    def get_value_prompt(self, state: str) -> str:
        return f"{state}\n\nThe value of this state is high"