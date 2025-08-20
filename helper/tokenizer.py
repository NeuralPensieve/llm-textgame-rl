import re
from typing import List
from transformers import AutoTokenizer
from config import PPOConfig


class TokenizerHelper:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_prompts(self, prompts: List[str]):
        """
        Tokenizes prompts with intelligent truncation of the history string for oversized prompts.
        """
        max_len = self.config.max_length
        
        truncated_prompts = []
        for prompt in prompts:
            # 1. First, tokenize the whole prompt to see if it needs truncation
            input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

            # 2. If it's too long, start trimming the history
            while len(input_ids) > max_len:
                # Find the history string using regex
                match = re.search(r"HISTORY: (.*?)\nSTATE:", prompt, re.DOTALL)
                if not match:
                    raise ValueError('State too long, and there is no HISTORY to cut')

                history_str = match.group(1).strip()
                
                # Split history into chunks and remove the oldest (the first one)
                history_chunks = [chunk.strip() for chunk in history_str.split('|')]
                
                if len(history_chunks) > 1:
                    # Remove the oldest history entry
                    new_history = " | ".join(history_chunks[1:])
                else:
                    # If only one history chunk left, we can't shorten it further.
                    # To prevent an infinite loop, we clear it.
                    new_history = "None"

                # Rebuild the prompt with the shorter history
                start_index = match.start(1)
                end_index = match.end(1)
                prompt = prompt[:start_index] + new_history + prompt[end_index:]

                # Re-tokenize to check the new length
                input_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            
            truncated_prompts.append(prompt)

        # 3. Now, tokenize the (potentially truncated) prompts together with padding
        padded = self.tokenizer(
            truncated_prompts,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return padded