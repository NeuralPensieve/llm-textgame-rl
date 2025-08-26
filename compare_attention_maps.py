import torch
import os
from datetime import datetime
import sys
import numpy as np

# Add the current directory to the path to import local modules
sys.path.append('.')

# Import the visualization functions
from helper.visualize_attention import get_all_layers_attention, visualize_model_comparison
from helper.create_video import create_video_from_pngs

# Import necessary classes for loading the checkpoint
from models import LLMPolicy
from helper import TokenizerHelper


def get_attention_from_loaded_model(model, tokenizer, prompt, model_name_for_log):
    """
    Get attention weights from an already loaded model
    """
    print(f"Getting attention weights for {model_name_for_log}...")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Clean tokens (from visualize_attention.py)
    def clean_tokens(tokens):
        cleaned = []
        for token in tokens:
            if token.startswith('Ġ'):
                processed_token = token[1:]
            elif token.startswith(' '):
                processed_token = token[1:]
            else:
                processed_token = token
            processed_token = processed_token.replace('Ċ', '\n')
            cleaned.append(processed_token)
        return cleaned
    
    tokens = clean_tokens(raw_tokens)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass with attention output
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention weights for all layers
    all_attentions = []
    for layer_idx, layer_attention in enumerate(outputs.attentions):
        # Get attention from the last token position (predicting next token)
        next_token_attention = layer_attention[0, :, -1, :].cpu().numpy()  # [heads, seq_len]
        all_attentions.append(next_token_attention)
    
    # Get the predicted next token
    next_token_logits = outputs.logits[0, -1, :]
    predicted_token_id = torch.argmax(next_token_logits).item()
    raw_predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
    predicted_token = clean_tokens([raw_predicted_token])[0]
    
    print(f"Predicted next token for {model_name_for_log}: '{predicted_token}'")
    print(f"Total layers: {len(all_attentions)}")
    
    return all_attentions, tokens, predicted_token, raw_tokens


def compare_attention_maps(prompt, checkpoint_path):
    """
    Compare attention maps between pretrained and fine-tuned models
    """
    
    # 1. Get attention data for the original pretrained model
    print("="*80)
    print("GETTING ATTENTION DATA FOR PRETRAINED MODEL")
    print("="*80)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    pretrained_attentions, tokens, pretrained_predicted, raw_tokens = get_all_layers_attention(model_name, prompt)
    
    # 2. Load the checkpoint and get attention data for the fine-tuned model
    print("\n" + "="*80)
    print("LOADING CHECKPOINT AND GETTING ATTENTION DATA FOR FINE-TUNED MODEL")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        checkpoints_dir = "checkpoints"
        if os.path.exists(checkpoints_dir):
            for f in os.listdir(checkpoints_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load the checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get the config from the checkpoint
    config = checkpoint["config"]
    print(f"Loaded checkpoint from iteration: {checkpoint['iteration']}")
    print(f"Model name: {config.model_name}")
    print(f"LoRA enabled: {config.lora_enabled}")
    print(f"Temperature at checkpoint: {checkpoint.get('temperature', 'N/A')}")
    
    # Create the tokenizer helper
    tokenizer_helper = TokenizerHelper(config)
    
    # Create the LLMPolicy with the same config
    print("Creating LLMPolicy...")
    policy = LLMPolicy(config, tokenizer_helper, device)
    
    # Load the state dict
    print("Loading model state dict...")
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    
    # Extract the underlying transformer model
    print(f"Policy model type: {type(policy.model)}")
    
    # Since LoRA is disabled, policy.model should be the full causal LM model
    transformer_model = policy.model
    print("Using policy.model directly (no LoRA)")
    
    print(f"Extracted transformer model type: {type(transformer_model)}")
    
    # Ensure the model is on the correct device and has the right settings for attention visualization
    transformer_model = transformer_model.to(device)
    
    # Fix Flash Attention 2 issue - we need to change to eager attention to output attention weights
    if hasattr(transformer_model.config, '_attn_implementation'):
        print(f"Current attention implementation: {transformer_model.config._attn_implementation}")
        transformer_model.config._attn_implementation = 'eager'
        print("Changed attention implementation to 'eager'")
    
    # We need to ensure output_attentions is enabled
    if hasattr(transformer_model.config, 'output_attentions'):
        transformer_model.config.output_attentions = True
        print("Enabled output_attentions")
    
    # Get attention data from fine-tuned model
    finetuned_attentions, tokens_ft, finetuned_predicted, _ = get_attention_from_loaded_model(
        transformer_model, 
        tokenizer_helper.tokenizer, 
        prompt,
        "Fine-tuned"
    )
    
    # Verify tokens match (they should be the same since it's the same prompt and tokenizer)
    if tokens != tokens_ft:
        print("WARNING: Token mismatch between models!")
        print(f"Pretrained tokens: {len(tokens)}")
        print(f"Fine-tuned tokens: {len(tokens_ft)}")
    
    # 3. Create comparison visualizations using the new function
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    # Control the gap between pretrained (top) and fine-tuned (bottom) visualizations
    # 0.5 = very tight, 1.0 = normal, 2.0 = loose, 0.2 = extremely tight
    gap_between_models = 0.0  # Change this value to adjust the gap!
    
    comparison_folder, _ = visualize_model_comparison(
        pretrained_attentions,
        finetuned_attentions, 
        tokens,
        pretrained_predicted,
        finetuned_predicted,
        "Pretrained",
        "Fine-tuned",
        raw_tokens,
        gap_between_models=gap_between_models
    )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"Comparison visualizations saved to: {comparison_folder}")
    print(f"Prompt used: '{prompt[:100]}...'")
    print(f"Pretrained predicted: '{pretrained_predicted}'")
    print(f"Fine-tuned predicted:  '{finetuned_predicted}'")
    print(f"\nEach image shows:")
    print(f"  - TOP: Pretrained model attention")
    print(f"  - BOTTOM: Fine-tuned model attention")
    print("Look for differences in attention patterns after PPO training!")

    return comparison_folder


if __name__ == "__main__":
    # The prompt to analyze
    # prompt = "You do! Here is how to play! First stop, head north. Then, attempt to venture east. Okay, and then, make absolutely sure that the locker inside the workshop is shut.\nBefore: None\nNow: You're now in the chamber.\nYou don't like doors? Why not try going north, that entranceway is unguarded.\nYou are carrying nothing.\n>"
    prompt = """Here is your task for today. First off, open the crate in the vault. After that, pick up the shadfly from the crate within the vault. Having got the shadfly, rest the shadfly on the rack. Got that?
Before: None
Now: You arrive in a vault. An ordinary one.
You make out a closed crate. You make out a rack. The rack is usual. Looks like someone's already been here and taken everything off it, though. Hm. Oh well
You need an unblocked exit? You should try going north. You don't like doors? Why not try going west, that entranceway is unguarded.
You are carrying nothing.
>"""
    # checkpoint_path = "checkpoints/best-medium-noRept/run_2025-08-24_08-37-48_iter_699.pt"
    checkpoint_path = "checkpoints/run_2025-08-25_21-29-15_iter_149.pt"

    comparison_folder = compare_attention_maps(prompt, checkpoint_path)

    create_video_from_pngs(
        input_folder = comparison_folder, 
        output_video=f"{comparison_folder}/comparison.mp4", 
        fps=1, 
        crop_top_percent=80
        )