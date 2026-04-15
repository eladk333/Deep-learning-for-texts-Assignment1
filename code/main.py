from __future__ import annotations
import os

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
    data_path = "../data/en/"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    weight_decay = 0.1

    num_batches_to_train = 50000
    checkpoint_every = 1000
    checkpoint_path = "checkpoints_en"
    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer, tokenized_data = data.load_data(data_path)
    
    # Calculate the 90% split index
    split_idx = int(len(tokenized_data) * 0.9)
    train_tokens = tokenized_data[:split_idx]
    val_tokens = tokenized_data[split_idx:]
    
    # 3. Chunk into large logical blocks
    chunk_size = 10000 
    
    train_data = [
        train_tokens[i : i + chunk_size] 
        for i in range(0, len(train_tokens), chunk_size)
    ]
    
    val_data = [
        val_tokens[i : i + chunk_size] 
        for i in range(0, len(val_tokens), chunk_size)
    ]
    
    print(f"Total tokens: {len(tokenized_data)}.")
    print(f"Created {len(train_data)} training blocks and {len(val_data)} validation blocks (Max block size: {chunk_size} tokens).")


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
    best_val_loss = float('inf') # Track the best loss for early stopping

    while True:
        # ---> 1. RECREATE TRAIN ITERATOR HERE (Starts a new Epoch) <---
        train_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))
        
        for batch in data.batch_items(train_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break
            
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            model.zero_grad() 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping) 
            optimizer.step() 

            num_batches += 1
            
            # Print training loss frequently
            if num_batches % 100 == 0:
                print(f"Seen {num_batches} batches. Train loss is: {loss.item():.4f}")
                
            # ---> EVALUATION BLOCK <---
            if num_batches % 500 == 0:
                model.eval() # Switch to eval mode
                val_loss_sum = 0.0
                eval_batches = 50 
                
                # ---> 2. RECREATE VAL ITERATOR HERE (Reset validation data) <---
                val_iter = iter(data.RandomOrderDataIterator(val_data, seq_len + 1))
                
                # Track how many batches we actually evaluate in case val data is small
                actual_eval_batches = 0 
                
                with torch.no_grad(): 
                    for i, val_batch in enumerate(data.batch_items(val_iter, batch_size)):
                        if i >= eval_batches:
                            break
                        v_batch_x, v_batch_y = lm.batch_to_labeled_samples(val_batch)
                        v_batch_x, v_batch_y = v_batch_x.to(device), v_batch_y.to(device)
                        
                        v_logits = model(v_batch_x)
                        v_loss = lm.compute_loss(v_logits, v_batch_y)
                        val_loss_sum += v_loss.item()
                        actual_eval_batches += 1
                
                avg_val_loss = val_loss_sum / actual_eval_batches
                print(f"\n--- Validation at Batch {num_batches} ---")
                print(f"Average Val Loss: {avg_val_loss:.4f}")
                
                # Sample text to see generation quality
                sampled = tokenizer.detokenize(
                    model.better_sample_continuation(tokenizer.tokenize("Romeo"), 50, temperature=0.8, topK=5)
                )
                print(f"Model sample: '''{sampled}'''\n")
                
                # Save only if it's the best validation loss we've seen
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'num_batches': num_batches,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                    }, f"{checkpoint_path}/best_model.pt")
                    print(f"*** New best model saved! Val Loss: {best_val_loss:.4f} ***")
                
                model.train() # Switch back to training mode
                
        if num_batches >= num_batches_to_train:
            break