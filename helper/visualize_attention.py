import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_tokens(tokens):
    """
    Clean up tokenizer artifacts like Ġ (space tokens) for better display
    and standardize newline characters.
    """
    cleaned = []
    for token in tokens:
        # First, remove space markers like Ġ
        if token.startswith('Ġ'):
            processed_token = token[1:]
        elif token.startswith(' '):
            processed_token = token[1:]
        else:
            processed_token = token
            
        # Now, replace newline markers like Ċ with a standard \n
        # This handles cases like '.Ċ' becoming '.\n'
        processed_token = processed_token.replace('Ċ', '\n')
        
        cleaned.append(processed_token)
    return cleaned

def get_all_layers_attention(model_name, prompt):
    """
    Get attention weights for all layers when predicting the next token
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
        attn_implementation="eager",
        output_attentions=True
    )
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    tokens = clean_tokens(raw_tokens)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
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
    
    return all_attentions, tokens, predicted_token, raw_tokens

def reconstruct_word_boundaries(tokens):
    """
    Reconstruct word boundaries to merge subword tokens into complete words
    while preserving token-level information for attention mapping
    """
    words = []
    word_to_tokens = []  # Maps each word to its constituent token indices
    current_word = ""
    current_token_indices = []
    
    for i, token in enumerate(tokens):
        if i == 0:
            # First token always starts a new word
            current_word = token
            current_token_indices = [i]
        elif token.startswith(('.', ',', '!', '?', ';', ':', "'", '"', ')', ']', '}')):
            # Punctuation - attach to current word then end it
            current_word += token
            current_token_indices.append(i)
            # End current word
            words.append(current_word)
            word_to_tokens.append(current_token_indices)
            current_word = ""
            current_token_indices = []
        elif token[0].isupper() and current_word:
            # New word starting with capital (but not first token)
            words.append(current_word)
            word_to_tokens.append(current_token_indices)
            current_word = token
            current_token_indices = [i]
        else:
            # Continue current word or start new word if current is empty
            if current_word:
                current_word += token
                current_token_indices.append(i)
            else:
                current_word = token
                current_token_indices = [i]
    
    # Add the last word if it exists
    if current_word:
        words.append(current_word)
        word_to_tokens.append(current_token_indices)
    
    return words, word_to_tokens

def create_simplified_text_visualization(attention_weights, tokens, predicted_token, 
                                       layer_idx, save_folder, raw_tokens, head_idx=None):
    """
    Create a simplified text visualization with only colored backgrounds
    """
    if head_idx is None:
        attention = np.mean(attention_weights, axis=0)
        title_suffix = "Average"
        filename_suffix = "avg"
    else:
        attention = attention_weights[head_idx]
        title_suffix = f"Head {head_idx}"
        filename_suffix = f"head_{head_idx}"
    
    # Normalize attention weights to 0-1 range
    attention_norm = (attention - attention.min()) / (attention.max() - attention.min())
    
    # Count newlines to determine figure height
    newline_count = sum(token.count('\n') for token in tokens)
    base_height = 6
    additional_height = newline_count * 0.6  # Extra height per newline
    
    # Create figure with adequate space
    fig, ax = plt.subplots(1, 1, figsize=(20, base_height + additional_height))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5 + additional_height)
    ax.axis('off')

    # Get the renderer to accurately measure text dimensions
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    # Create colormap (white to red)
    colors = ['white', 'lightcoral', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=100)
    
    # Title
    ax.text(10, 4.5 + additional_height, f'Layer {layer_idx} ({title_suffix}) → "{predicted_token}"', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Layout text with clear separation between tokens
    x_pos = 0.5
    y_pos = 3.8 + additional_height
    # char_width = 0.09  # This will be replaced by dynamic calculation
    token_gap = 0.15   # Gap between token rectangles
    max_line_width = 19.5 # Adjusted max line width
    current_line_width = 0
    line_height = 0.5
    
    for i, (token, token_attention) in enumerate(zip(tokens, attention_norm)):
        # A single token can contain both text and newlines, e.g., ".\n"
        # We process the text part first, then all the newlines.
        
        # Check if the token contains a newline and split it
        if '\n' in token:
            display_part = token.split('\n')[0] # Text before the first newline
            newline_count = token.count('\n')
        else:
            display_part = token
            newline_count = 0

        # --- 1. Render the displayable text part (if it exists) ---
        if display_part:
            # Dynamically calculate token width
            text_object = ax.text(0, 0, display_part, fontsize=10, fontweight='bold', ha='left', va='center')
            text_width_display_units = text_object.get_window_extent(renderer=renderer).width / fig.dpi * (20 / fig.get_size_inches()[0])
            text_object.remove() # Remove the temporary text object

            token_padding = 0.1 # Small padding around the text
            token_width = text_width_display_units + token_padding

            total_width_needed = token_width + (token_gap if current_line_width > 0 else 0)
            
            if current_line_width + total_width_needed > max_line_width and current_line_width > 0.5: # Ensure a meaningful amount of text is on the line before wrapping
                y_pos -= line_height
                x_pos = 0.5
                current_line_width = 0
            
            if current_line_width > 0:
                x_pos += token_gap
                current_line_width += token_gap
            
            color = cmap(token_attention)
            
            rect = patches.Rectangle((x_pos, y_pos - 0.12), 
                                   token_width, 0.24,
                                   facecolor=color, edgecolor='gray', linewidth=0.3,
                                   alpha=0.9)
            ax.add_patch(rect)
            
            ax.text(x_pos + token_width/2, y_pos, display_part, 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='black')
            
            x_pos += token_width
            current_line_width += token_width

        # --- 2. Handle any newlines contained in the token ---
        if newline_count > 0:
            y_pos -= (line_height * newline_count)
            x_pos = 0.5
            current_line_width = 0
    
    # Add minimal colorbar at bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=attention.min(), vmax=attention.max()))
    sm.set_array([])
    
    # Create colorbar at bottom
    cbar_ax = fig.add_axes([0.1, 0.08, 0.8, 0.03])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Attention Weight', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.18, top=0.92)
    
    # Save the figure
    if isinstance(layer_idx, int):
        filename = f'layer_{layer_idx:02d}_{filename_suffix}.png'
    else:
        filename = f'layer_{layer_idx}_{filename_suffix}.png'
    
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()  # Close to save memory
    
    return save_path

def visualize_all_layers(model_name, prompt):
    """
    Create visualizations for all layers and save to timestamped folder
    """
    # Create timestamped folder
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    save_folder = f'visualizations/{timestamp}'
    os.makedirs(save_folder, exist_ok=True)
    
    print(f"Getting attention weights for all layers...")
    all_attentions, tokens, predicted_token, raw_tokens = get_all_layers_attention(model_name, prompt)
    
    print(f"Model: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Raw tokens: {raw_tokens}")
    # Use repr() to make newlines visible in the console output
    print(f"Cleaned tokens: {[repr(t) for t in tokens]}")
    print(f"Predicted next token: '{predicted_token}'")
    print(f"Total layers: {len(all_attentions)}")
    print(f"Saving to folder: {save_folder}")
    
    # Debug: Print which tokens are detected as newlines
    newline_indices = []
    for i, token in enumerate(tokens):
        if '\n' in token:
            newline_indices.append(i)
            # Use repr() on token to clearly show the \n character
            print(f"  Newline detected at index {i}: '{raw_tokens[i]}' -> {repr(token)}")
    
    print(f"Found {len(newline_indices)} newline tokens")
    print()
    
    saved_files = []
    
    # Create visualization for each layer (average across heads)
    for layer_idx, attention_weights in enumerate(all_attentions):
        print(f"Processing layer {layer_idx + 1}/{len(all_attentions)}...", end=' ')
        
        save_path = create_simplified_text_visualization(
            attention_weights, tokens, predicted_token, 
            layer_idx, save_folder, raw_tokens
        )
        saved_files.append(save_path)
        print(f"✓ Saved: {os.path.basename(save_path)}")
    
    # Create overall average across all layers
    print("Creating overall average...", end=' ')
    all_layers_avg = np.mean([np.mean(att, axis=0) for att in all_attentions], axis=0)
    all_layers_avg = all_layers_avg.reshape(1, -1)  # Reshape to match expected format
    
    save_path = create_simplified_text_visualization(
        all_layers_avg, tokens, predicted_token, 
        "ALL", save_folder, raw_tokens
    )
    saved_files.append(save_path)
    print(f"✓ Saved: {os.path.basename(save_path)}")
    
    print(f"\n✅ Complete! Saved {len(saved_files)} visualizations to: {save_folder}")
    return save_folder, saved_files

def create_comparison_text_visualization(attention_weights_1, attention_weights_2, tokens, 
                                       predicted_token_1, predicted_token_2, 
                                       model_name_1, model_name_2,
                                       layer_idx, save_folder, raw_tokens, head_idx=None, gap_between_models=1.0):
    """
    Create a comparison visualization with two models (top and bottom)
    
    Args:
        gap_between_models: Vertical space between the two model visualizations in plot units.
                           0.0 = touching, 1.0 = gap equal to text height, 2.0 = gap twice the text height
    """
    if head_idx is None:
        attention_1 = np.mean(attention_weights_1, axis=0)
        attention_2 = np.mean(attention_weights_2, axis=0)
        title_suffix = "Average"
        filename_suffix = "avg"
    else:
        attention_1 = attention_weights_1[head_idx]
        attention_2 = attention_weights_2[head_idx]
        title_suffix = f"Head {head_idx}"
        filename_suffix = f"head_{head_idx}"
    
    # Normalize attention weights to 0-1 range
    attention_1_norm = (attention_1 - attention_1.min()) / (attention_1.max() - attention_1.min())
    attention_2_norm = (attention_2 - attention_2.min()) / (attention_2.max() - attention_2.min())
    
    # Count newlines to determine text height needed
    newline_count = sum(token.count('\n') for token in tokens)
    base_content_height = 3.0  # Approximate height needed for text content
    additional_height = newline_count * 0.6
    content_height = base_content_height + additional_height
    
    # Calculate precise spacing
    title_height = 0.5
    margin_top = 0.5
    margin_bottom = 1.0  # Extra space for colorbar
    
    single_model_height = title_height + content_height
    total_content_height = single_model_height * 2 + (gap_between_models * single_model_height)
    total_figure_height = total_content_height + margin_top + margin_bottom
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, total_figure_height))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, total_figure_height)
    ax.axis('off')

    # Get the renderer
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    # Create colormap
    colors = ['white', 'lightcoral', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=100)
    
    # Calculate precise y-positions for each model
    # Model 2 (bottom) starts from the bottom
    model2_bottom = margin_bottom
    model2_top = model2_bottom + single_model_height
    
    # Model 1 (top) starts after model 2 + gap
    model1_bottom = model2_top + (gap_between_models * single_model_height)
    model1_top = model1_bottom + single_model_height
    
    # Function to render text with attention at specific y bounds
    def render_model_attention(attention_norm, predicted_token, model_name, bottom_y, top_y):
        # Title at the top of the allocated space
        title_y = top_y - title_height/2
        ax.text(10, title_y, f'{model_name} - Layer {layer_idx} ({title_suffix}) → "{predicted_token}"', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Text content starts below title and works downward
        x_pos = 0.5
        y_pos = top_y - title_height - 0.2  # Start just below title with small margin
        token_gap = 0.15
        max_line_width = 19.5
        current_line_width = 0
        line_height = 0.5
        
        for i, (token, token_attention) in enumerate(zip(tokens, attention_norm)):
            # Handle newlines and display part
            if '\n' in token:
                display_part = token.split('\n')[0]
                newline_count_token = token.count('\n')
            else:
                display_part = token
                newline_count_token = 0

            # Render displayable text
            if display_part:
                # Calculate token width
                text_object = ax.text(0, 0, display_part, fontsize=10, fontweight='bold', ha='left', va='center')
                text_width_display_units = text_object.get_window_extent(renderer=renderer).width / fig.dpi * (20 / fig.get_size_inches()[0])
                text_object.remove()

                token_padding = 0.1
                token_width = text_width_display_units + token_padding
                total_width_needed = token_width + (token_gap if current_line_width > 0 else 0)
                
                # Line wrapping
                if current_line_width + total_width_needed > max_line_width and current_line_width > 0.5:
                    y_pos -= line_height
                    x_pos = 0.5
                    current_line_width = 0
                
                if current_line_width > 0:
                    x_pos += token_gap
                    current_line_width += token_gap
                
                color = cmap(token_attention)
                
                rect = patches.Rectangle((x_pos, y_pos - 0.12), 
                                       token_width, 0.24,
                                       facecolor=color, edgecolor='gray', linewidth=0.3,
                                       alpha=0.9)
                ax.add_patch(rect)
                
                ax.text(x_pos + token_width/2, y_pos, display_part, 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='black')
                
                x_pos += token_width
                current_line_width += token_width

            # Handle newlines
            if newline_count_token > 0:
                y_pos -= (line_height * newline_count_token)
                x_pos = 0.5
                current_line_width = 0
    
    # Render both models with precise positioning
    render_model_attention(attention_1_norm, predicted_token_1, model_name_1, model1_bottom, model1_top)
    render_model_attention(attention_2_norm, predicted_token_2, model_name_2, model2_bottom, model2_top)
    
    # Add colorbar at bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    cbar_height = 0.03
    cbar_margin = 0.02
    cbar_ax = fig.add_axes([0.1, cbar_margin/total_figure_height, 0.8, cbar_height])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Attention Weight (Normalized)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.08, top=0.98)
    
    # Save the figure
    if isinstance(layer_idx, int):
        filename = f'comparison_layer_{layer_idx:02d}_{filename_suffix}.png'
    else:
        filename = f'comparison_layer_{layer_idx}_{filename_suffix}.png'
    
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()
    
    return save_path


def visualize_model_comparison(model1_attentions, model2_attentions, tokens, 
                              predicted_token_1, predicted_token_2, 
                              model_name_1, model_name_2, raw_tokens, gap_between_models=0.02):
    """
    Create comparison visualizations for all layers between two models
    
    Args:
        gap_between_models: Controls the vertical gap between pretrained and fine-tuned visualizations.
                          0.0 = no gap, 0.02 = small gap, 0.1 = large gap
    """
    # Create timestamped folder
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    save_folder = f'visualizations/{timestamp}_comparison'
    os.makedirs(save_folder, exist_ok=True)
    
    print(f"Creating comparison visualizations...")
    print(f"Model 1: {model_name_1}")
    print(f"Model 2: {model_name_2}")
    print(f"Predicted tokens: '{predicted_token_1}' vs '{predicted_token_2}'")
    print(f"Total layers: {len(model1_attentions)}")
    print(f"Saving to folder: {save_folder}")
    
    saved_files = []
    
    # Create visualization for each layer (average across heads)
    for layer_idx, (att1, att2) in enumerate(zip(model1_attentions, model2_attentions)):
        print(f"Processing layer {layer_idx + 1}/{len(model1_attentions)}...", end=' ')
        
        save_path = create_comparison_text_visualization(
            att1, att2, tokens, predicted_token_1, predicted_token_2,
            model_name_1, model_name_2, layer_idx, save_folder, raw_tokens, 
            head_idx=None, gap_between_models=gap_between_models
        )
        saved_files.append(save_path)
        print(f"✓ Saved: {os.path.basename(save_path)}")
    
    # Create overall average across all layers
    # print("Creating overall average...", end=' ')
    # model1_avg = np.mean([np.mean(att, axis=0) for att in model1_attentions], axis=0)
    # model2_avg = np.mean([np.mean(att, axis=0) for att in model2_attentions], axis=0)
    # model1_avg = model1_avg.reshape(1, -1)
    # model2_avg = model2_avg.reshape(1, -1)
    
    # save_path = create_comparison_text_visualization(
    #     model1_avg, model2_avg, tokens, predicted_token_1, predicted_token_2,
    #     model_name_1, model_name_2, "ALL", save_folder, raw_tokens,
    #     head_idx=None, gap_between_models=gap_between_models
    # )
    # saved_files.append(save_path)
    # print(f"✓ Saved: {os.path.basename(save_path)}")
    
    print(f"\n✅ Complete! Saved {len(saved_files)} comparison visualizations to: {save_folder}")
    return save_folder, saved_files