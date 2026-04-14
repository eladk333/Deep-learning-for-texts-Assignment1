from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    weight_decay = 0.1

    num_batches_to_train = 50000
    checkpoint_every = 1000
    checkpoint_path = "checkpoints_he"
    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer, tokenized_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

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
    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break
            #num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # The apple is very pretty
            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            model.zero_grad() # Cleaning the gradient of parameters
            loss.backward() # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping) # cliping to avoid exploding gradients
            optimizer.step() # Updating the parameters with the optimizer the 'strategy to walk'

            num_batches += 1
            if num_batches % 100 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 1000 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(
                            model.better_sample_continuation(tokenizer.tokenize("Hello"), 500, temperature=0.8, topK=5)
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''")

                # Checkpoint save
                if num_batches % checkpoint_every == 0:
                    torch.save({
                        'num_batches': num_batches,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, f"{checkpoint_path}/checkpoint_{num_batches}.pt")
                    print(f"Saved checkpoint at batch {num_batches}")
                    print("")
