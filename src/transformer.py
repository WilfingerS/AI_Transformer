import torch
import torch.nn as nn
from fileutils import loadData
from torch.nn import functional as F



class Transformer():
    def __init__(self):
        super().__init__()
        # hyperparameters
        data = loadData()
        print(data)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # ------------
        torch.manual_seed(data["seed"])

        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open(data["input_file"], 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # Train and test splits
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val
        train_data = data[:n]
        val_data = data[n:]

        # data loading
        def get_batch(split):
            # generate a small batch of data of inputs x and targets y
            data = train_data if split == 'train' else val_data
            ix = torch.randint(len(data) - data["block_size"], (data["batch_size"],))
            x = torch.stack([data[i:i+data["block_size"]] for i in ix])
            y = torch.stack([data[i+1:i+data["block_size"]+1] for i in ix])
            x, y = x.to(device), y.to(device)
            return x, y

        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(data["eval_iters"])
                for k in range(data["eval_iters"]):
                    X, Y = get_batch(split)
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out




        from helper import BigramLanguageModel
        model = BigramLanguageModel(vocab_size)
        m = model.to(device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        # create a PyTorch optimizer
        print(data["learning_rate"])
        learning_rate = data["learning_rate"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for iter in range(data["max_iters"]):

            # every once in a while evaluate the loss on train and val sets
            if iter % data["eval_interval"] == 0 or iter == data["max_iters"] - 1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = get_batch('train')

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # generate from the model
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))