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
    W : torch.Tensor

def init_nn():
    W = torch.randn((27,27), requires_grad=True)
    return NN(W)


def init_context():
    with open("datasets/names.txt", 'r') as f:
        words = f.read().splitlines()
    chs = sorted(set(list(''.join(words))))

    stoi = {c : i+1 for i,c in enumerate(chs)}
    stoi['.'] = 0
    itos = {v : k for k,v in stoi.items()}
    ctx = Context(words, chs, stoi, itos)
    return ctx


def dataset(ctx):
    X, Y = [], []
    for w in ctx.words:
        w = '.' + w + '.'
        for i in range(len(w) - 1):
            ix = ctx.stoi[w[i]]
            iy = ctx.stoi[w[i+1]]
            X.append(ix)
            Y.append(iy)
    return torch.tensor(X), torch.tensor(Y)


def SGD(nn: NN, context, X : torch.Tensor, Y : torch.Tensor, lr= 0.1, passes = 10):
    xenc = F.one_hot(X, num_classes=27).float()
    #print(xenc.shape)
    #print(nn.W.shape) 
    num = X.nelement()
    for k in range(passes):
        logits = xenc @ nn.W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(num), Y].log().mean()

        nn.W.grad = None
        loss.backward()
        clr = lr if k < passes / 2 else lr / 8

        nn.W.data += -clr * nn.W.grad
    print("loss after SGD:", loss.item())

def generate(ctx : Context, nn : NN, nitems):
    res = []
    for i in range(nitems):
        w = []
        ix = ctx.stoi['.']
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ nn.W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)
            ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
            ch = ctx.itos[ix]
            if ch == '.':
                break
            w.append(ch)
        res.append(''.join(w))
    return res


def main():
    ctx = init_context()
    nn = init_nn()
    X,Y = dataset(ctx)
    SGD(nn, ctx, X, Y, lr = 20.0, passes=500)

    r = generate(ctx, nn, 30)
    print(r)

if __name__ == "__main__":
    main()