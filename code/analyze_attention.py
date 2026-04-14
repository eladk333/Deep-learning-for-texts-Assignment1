import torch
import data
from transformer import TransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the tokenizer (we don't need the dataset iterator here)
data_path = "../data/en/"
tokenizer, _ = data.load_data(data_path)

# 2. Recreate the model with the exact same parameters as main.py
model = TransformerLM(
    n_layers=6,
    n_heads=6,
    embed_size=192,
    max_context_len=128,
    vocab_size=tokenizer.vocab_size(),
    mlp_hidden_size=192 * 4,
    with_residuals=True,
).to(device)

# 3. Load your trained checkpoint
checkpoint_path = "checkpoints_en/checkpoint_50000.pt" # Change to your latest checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 4. Prepare your input text
text_to_analyze = "The apple is very pretty"
# Tokenize and convert to a batch of size (1, N)
tokens = tokenizer.tokenize(text_to_analyze)
input_batch = torch.tensor([tokens], dtype=torch.int32).to(device)

# 5. Run the code snippet!
model.eval()
with torch.no_grad():
    # Pass in your tokenized input sequence
    logits, all_attention = model(input_batch, return_attention=True)

# all_attention is a list of tensors containing your raw attention distributions.
print(f"Captured {len(all_attention)} attention matrices.")
print(f"Shape of the first matrix: {all_attention[0].shape}")

# Example: Inspecting Layer 1, Head 1
layer1_head1_attention = all_attention[0]
print(f"Attention weights for '{text_to_analyze}':\n", layer1_head1_attention)