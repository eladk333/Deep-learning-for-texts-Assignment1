from __future__ import annotations
import os
import random

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    import lm
    import torch
    from torch import nn, optim
    from transformer import TransformerLM

    import data

    seq_len = 128
    batch_size = 64
    data_path = "../data/he/"
    n_layers = 10
    n_heads = 10
    embed_size = 100
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    weight_decay = 0.1

    num_batches_to_train = 20000
    checkpoint_every = 1000
    checkpoint_path = "he_checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer, tokenized_data = data.load_data(data_path)

    # --- 1. CHUNK AND RANDOMIZE EVERY 10,000 TOKENS ---
    chunk_size = 10000
    
    # Slice the massive token stream into 10k chunks directly
    chunks = [
        tokenized_data[i : i + chunk_size] 
        for i in range(0, len(tokenized_data), chunk_size) 
        if len(tokenized_data[i : i + chunk_size]) > seq_len
    ]
    
    random.shuffle(chunks)

    split_idx = int(len(chunks) * 0.9)
    train_chunks = chunks[:split_idx]
    val_chunks = chunks[split_idx:]
    
    print(f"Total chunks: {len(chunks)} | Train: {len(train_chunks)} | Validation: {len(val_chunks)}")

    # Initialize data iterators for both sets
    train_iter = iter(data.RandomOrderDataIterator(train_chunks, seq_len + 1))
    val_iter = iter(data.RandomOrderDataIterator(val_chunks, seq_len + 1))
    
    train_batch_iter = data.batch_items(train_iter, batch_size)
    val_batch_iter = data.batch_items(val_iter, batch_size)

    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
    ).to(device)
    
    # Uncomment to resume from checkpoint:
    # checkpoint = torch.load("checkpoints/checkpoint_1000.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # num_batches = checkpoint['num_batches']
    # print(f"Resumed from batch {num_batches}")
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and LayerNorm weights are 1D. Weight matrices and embeddings are >= 2D.
        if param.dim() < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = optim.AdamW(optim_groups, lr=learning_rate, betas=[0.9, 0.95])

    model.train()

    num_batches = 0
    lowest_val_loss = float('inf')
    lowest_val_loss_batch = 0

    for batch in train_batch_iter:
        if num_batches >= num_batches_to_train:
            break

        batch_x, batch_y = lm.batch_to_labeled_samples(batch)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = lm.compute_loss(logits, batch_y)

        # Parameters update
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        num_batches += 1

        if num_batches % 1000 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    val_batch = next(val_batch_iter)
                    v_batch_x, v_batch_y = lm.batch_to_labeled_samples(val_batch)
                    v_batch_x = v_batch_x.to(device)
                    v_batch_y = v_batch_y.to(device)
                    val_logits = model(v_batch_x)
                    val_loss = lm.compute_loss(val_logits, v_batch_y)
                    val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)

            if avg_val_loss < lowest_val_loss:
                lowest_val_loss = avg_val_loss
                lowest_val_loss_batch = num_batches

            print(f"Seen {num_batches} batches. Train loss: {loss.item():.4f} | avgVal loss : {avg_val_loss:.4f}")

            if num_batches % 10000 == 0:
                for _ in range(1):
                    sampled = tokenizer.detokenize(
                        model.better_sample_continuation(tokenizer.tokenize("romeo"), 500, temperature=0.8, topK=5)
                    )
                    print(f"Model sample: '''{sampled}'''")

            model.train()

            # Checkpoint save
            if num_batches % checkpoint_every == 0:
                torch.save({
                    'num_batches': num_batches,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f"{checkpoint_path}/checkpoint_{num_batches}.pt")
                print(f"Saved checkpoint at batch {num_batches}\n")

    print(f"\nLowest validation loss: {lowest_val_loss:.4f} at batch {lowest_val_loss_batch}")