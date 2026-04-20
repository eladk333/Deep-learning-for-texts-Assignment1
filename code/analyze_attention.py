import torch
import data
from transformer import TransformerLM
import math
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the tokenizer
data_path = "../data/en/"
tokenizer, _ = data.load_data(data_path)

# 2. Recreate the model with the exact same parameters as main.py
model = TransformerLM(
    n_layers=10,
    n_heads=10,
    embed_size=100,
    max_context_len=128,
    vocab_size=tokenizer.vocab_size(),
    mlp_hidden_size= 100 * 4,
    with_residuals=True,
).to(device)

# 3. Load your trained checkpoint
checkpoint_path = "checkpoints/checkpoint_20000.pt" # Change to your latest checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 4. Prepare your input text
text_to_analyze = "I, sir! ne'er a whit."
tokens = tokenizer.tokenize(text_to_analyze)
input_batch = torch.tensor([tokens], dtype=torch.int32).to(device)

# 5. Run the model to get attention weights
model.eval()
with torch.no_grad():
    logits, all_attention = model(input_batch, return_attention=True)

# 6. --- HELPER: Custom Token Label Decoding ---
def get_token_labels(tokenizer, tokens):
    labels = []
    for t in tokens:
        # In CharTokenizer, .vocab is a list where the index is the token ID
        char = tokenizer.vocab[int(t)]
        
        # We still use repr() so that a space doesn't just show up as an invisible blank on the graph
        readable_char = repr(char).strip("'").strip('"')
        labels.append(readable_char)
        
    return labels

token_labels = get_token_labels(tokenizer, tokens)

# 7. --- VISUALIZER: Grid for all heads ---
# 7. --- VISUALIZER: Bulletproof Grid for all heads ---

def extract_all_2d_matrices(attention_data):
    """Recursively digs through lists and tensors to find all 2D attention matrices."""
    matrices = []
    if isinstance(attention_data, list) or isinstance(attention_data, tuple):
        for item in attention_data:
            matrices.extend(extract_all_2d_matrices(item))
    elif isinstance(attention_data, torch.Tensor):
        # Remove any dimensions of size 1 (like batch=1)
        t = attention_data.squeeze().cpu().numpy()
        if t.ndim == 2:
            matrices.append(t)
        elif t.ndim == 3:
            # Shape: (num_heads, seq_len, seq_len)
            for i in range(t.shape[0]):
                matrices.append(t[i])
        elif t.ndim == 4:
            # Shape: (batch, num_heads, seq_len, seq_len)
            for b in range(t.shape[0]):
                for h in range(t.shape[1]):
                    matrices.append(t[b, h])
    return matrices

def plot_layer_heads(all_attention, labels, layer_idx, total_heads_per_layer=10):
    """
    Plots a grid of heatmaps for all attention heads in a specified layer.
    """
    # 1. Gather every single 2D matrix returned by the model
    all_matrices = extract_all_2d_matrices(all_attention)
    
    # 2. Slice out just the heads for the requested layer
    start_idx = layer_idx * total_heads_per_layer
    end_idx = start_idx + total_heads_per_layer
    
    layer_matrices = all_matrices[start_idx:end_idx]
    
    if len(layer_matrices) == 0:
        print(f"Error: Could not find matrices for layer {layer_idx}.")
        print(f"Total individual matrices captured: {len(all_matrices)}")
        return
        
    num_heads_to_plot = len(layer_matrices)
    matrix_size = layer_matrices[0].shape[0]

    # 3. Label safety check
    if len(labels) != matrix_size:
        if len(labels) > matrix_size:
            labels = labels[:matrix_size]
        else:
            labels = labels + ["[PAD]"] * (matrix_size - len(labels))

    # 4. Dynamically calculate grid dimensions (max 5 columns per row)
    cols = min(5, num_heads_to_plot)
    rows = math.ceil(num_heads_to_plot / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_heads_to_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 5. Plot each head
    for i, attn_matrix in enumerate(layer_matrices):
        ax = axes[i]
        sns.heatmap(
            attn_matrix, 
            annot=False,          
            cmap="viridis",      
            xticklabels=labels,  
            yticklabels=labels,  
            vmin=0.0,             
            vmax=1.0,             
            cbar=(i % cols == cols - 1),
            ax=ax
        )
        ax.set_title(f"Head {i}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # 6. Hide any extra empty subplots 
    for i in range(num_heads_to_plot, len(axes)):
        fig.delaxes(axes[i])

    # 7. Add main title and adjust layout
    fig.suptitle(f"Attention heads for Layer {layer_idx}, text: '{''.join(labels)}'", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    plt.show()

# Visualize all 10 heads for Layer 0 (the first layer)
# Notice we now pass the *entire* all_attention list, not just all_attention[0]
plot_layer_heads(all_attention, token_labels, layer_idx=0, total_heads_per_layer=10)