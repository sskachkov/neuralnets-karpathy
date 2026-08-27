from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F
import random

@dataclass
class Context:
    words : Any
    chs : Any
    stoi : Any
    itos : Any

@dataclass
class NN:
    C : torch.Tensor
    W1 : torch.Tensor
    b1 : torch.Tensor
    W2 : torch.Tensor
    b2 : torch.Tensor
    W1 : torch.Tensor
    params : torch.Tensor

@dataclass
class Dataset:
    Xtr : torch.tensor
    Ytr : torch.tensor
    Xdev : torch.tensor
    Ydev : torch.tensor
    Xtest : torch.tensor
    Ytest : torch.tensor

def init_context():
    with open("datasets/names.txt", 'r') as f:
        words = f.read().splitlines()
    chs = sorted(set(list(''.join(words))))

    stoi = {c : i+1 for i,c in enumerate(chs)}
    stoi['.'] = 0
    itos = {v : k for k,v in stoi.items()}
    ctx = Context(words, chs, stoi, itos)
    return ctx


def init_dataset(ctx : Context):
    def helper(ctx, words):
        X, Y = [], []
        for w in words:
            block = [0] * 3
            for i in range(len(w)):
                ix = ctx.stoi[w[i]]
                X.append(block)
                Y.append(ix)
                block = block[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X,Y
    words = list(ctx.words)
    random.shuffle(words)
    i1 = int(len(words) * 0.8)
    i2 = int(len(words) * 0.9)
    Xtr, Ytr = helper(ctx, words[:i1])
    Xdev, Ydev = helper(ctx, words[i1:i2])
    Xtest, Ytest = helper(ctx, words[i2:])
    return Dataset(Xtr, Ytr, Xdev, Ydev, Xtest, Ytest)

def init_nn():
    C = torch.randn(27, 20, requires_grad=True)
    # 10 dimensions per embedding means 30 per neuron, we will have 200 neurons
    W1 = torch.randn((60, 100), requires_grad=True)
    b1 = torch.randn((100), requires_grad=True)
    W2 = torch.randn(100, 27, requires_grad=True)
    b2 = torch.randn(27, requires_grad=True)
    params = [C, W1, b1, W2, b2]
    return NN(C, W1, b1, W2, b2, params)

def SGD(nn: NN, context, dataset : Dataset, lr= 0.1, passes = 1000, batch_size = 32):
    for k in range(passes):
        ix = torch.randint(0, dataset.Xtr.shape[0], (batch_size,))
        emb = nn.C[dataset.Xtr[ix]]
        #print(emb.shape)
        h1 = emb.view(-1, 60) @ nn.W1 + nn.b1
        #print(h1.shape)
        logits = h1 @ nn.W2 + nn.b2
        loss = F.cross_entropy(logits, dataset.Ytr[ix])

        for p in nn.params:
            p.grad = None

        loss.backward()
        
        if k < passes / 2:
            clr = lr
        elif k < passes / 1.7:
            clr = lr / 10
        elif k < passes / 1.5:
            clr = lr / 30
        elif k < passes / 1.3:
            clr = lr / 40
        else:
            clr = lr / 50
        for p in nn.params:
            p.data += -clr * p.grad
    print("Loss after SGD:", loss.item())
    # loss for dev data
    emb = nn.C[dataset.Xdev]
    #print(emb.shape)
    h1 = emb.view(-1, 60) @ nn.W1 + nn.b1
    #print(h1.shape)
    logits = h1 @ nn.W2 + nn.b2
    loss = F.cross_entropy(logits, dataset.Ydev)
    print("Loss on dev data:", loss.item())

def main():
    ctx = init_context()
    nn = init_nn()
    dataset = init_dataset(ctx)

    SGD(nn, ctx, dataset, lr = 0.01, passes=50000, batch_size=64)

    # r = generate(ctx, nn, 30)
    # print(r)

if __name__ == "__main__":
    main()