import torch
from typing import List
from transformers import AutoTokenizer

class TrieNode:
    def __init__(self, value: int):
        self.value = value
        self.n = 0
        self.children = {}


class Trie:
    eos_id = 0
    def __init__(self, start: int):
        self.head = TrieNode(start)

    def insert(self, token_ids: List[int]):
        curr = self.head
        for id in token_ids:
            if id not in curr.children:
                curr.children[id] = TrieNode(id)
            curr.n += 1
            curr = curr.children[id]
        
        if Trie.eos_id == 0:
            Trie.eos_id = token_ids[-1]
        else:
            if Trie.eos_id != token_ids[-1]:
                raise ValueError("There is a mismatch between EOS token IDs. Each action sequence must end with the EOS token.")
            else:
                Trie.eos_id = token_ids[-1]  # Update the EOS ID to the last token in the sequence which should be the EOS token

    def update_head(self, new_head: int):
        output = []
        for key in self.head.children.keys():
            if key == new_head:
                self.head = self.head.children[key]
                output.append(key)
                while key != Trie.eos_id and len(self.head.children) == 1:
                    self.head = next(iter(self.head.children.values()))
                    key = self.head.value
                    output.append(key)
                break

        if not output:
            raise ValueError(f"{new_head} not in head's children: {self.head.children.keys()}")
        
        return output
        
    def visualize(self, node=None, prefix="", level=0):
        """Recursively print the trie structure."""
        if node is None:
            node = self.head

        for id, child in node.children.items():
            marker = "*" if child.value == 1 else ""
            print("   " * level + f"└── {id}{marker}")
            self.visualize(child, f"{prefix}{id}, ", level + 1)


def generate_mask(tries: List[Trie], vocab_size: int, data_type=torch.float32) -> torch.Tensor:
    mask = torch.full((len(tries), vocab_size), float('-inf')).to(dtype=data_type)
    for i, trie in enumerate(tries):
        valid_tokens = list(trie.head.children.keys())
        if valid_tokens:
            mask[i, valid_tokens] = 0.0
    return mask


def tokenize_actions_for_trie(action_list: List[str], tokenizer: AutoTokenizer) -> List[List[int]]:
    """
    Tokenizes a list of action strings and appends the EOS token to each.
    
    Args:
        action_list: A list of available action strings (e.g., ["go north", "look"]).
        tokenizer: The Hugging Face tokenizer instance.
        
    Returns:
        A list of lists, where each inner list contains the token IDs for an action
        ending with the EOS token.
    """
    # Ensure the tokenizer has an EOS token defined.
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token.")
        
    tokenized_actions = []
    for action_str in action_list:
        # Tokenize the action, appending a space before the action string
        token_ids = tokenizer.encode(action_str, add_special_tokens=False)
        # Manually append the EOS token ID
        token_ids.append(tokenizer.eos_token_id)
        tokenized_actions.append(token_ids)
        
    return tokenized_actions