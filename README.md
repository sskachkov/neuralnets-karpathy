# Neural Networks: Zero to Hero

Working through Andrej Karpathy's [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) — building neural nets from raw gradients up through a GPT-style transformer.

## Project layout
* [`01-micrograd/`](01-micrograd) From the backpropagation lecture. `Value`-based autograd engine (`+`, `*`, `**`, `tanh`, `exp`, topo-sort backward pass) + a small MLP (pure Python, no tensors). Gradients cross-checked against PyTorch's own autograd on the same computation.
* [`02-makemore/`](02-makemore) From makemore lectures, parts 1–3. Character-level name generator: bigram counts → single-layer NN → MLP w/ embeddings → deeper MLP w/ BatchNorm + init diagnostics.
* [`03-gpt/`](03-gpt) From the Let's build GPT lecture. Decoder-only transformer (multi-head self-attention, feedforward, residuals, layernorm), char-level, trained on Shakespeare.

## Running things

```sh
uv sync
source .venv/bin/activate
```
Then just run any `.py` file, for ex: `python3 03-gpt/attention.py`.
To open and run notebooks use a Jupyter-compatible editor.

`01-micrograd` requires native graphviz library, so `brew install graphviz` might be required.

Notebooks and scripts read from `datasets/` — see the comment at the top of each file for where that particular dataset came from.

**Gotcha:** a `.py` script run from the repo root resolves `datasets/...` fine, but a notebook's kernel CWD defaults to *the notebook's own folder*, not the repo root — so from inside `02-makemore/`, the same path needs to be `../datasets/...`. 

## Notes 

- `01-micrograd` is pure Python/scalar — no tensors, no matmul, unlike the torch-based `02-makemore`/`03-gpt`. This is deliberate, so every step of the computation graph stays observable.
- **MPS can be *slower* than CPU** on `03-gpt` at small batch sizes (e.g. `batch_size=64`) — MPS has fixed per-op dispatch overhead that dominates when tensors are tiny. Bumping `batch_size` up (256 worked well) amortizes that overhead and flips MPS ahead of CPU. 
- **`03-gpt/attention.py` has no checkpointing.** A training run has to complete in one sitting — killing it partway loses everything. Worth remembering before kicking off a long run.
- **`03-gpt/attention.py`**  plateaus at ~1.50 loss; further tuning seems to yield little benefit. Next logical steps: a tokenizer and a larger training set.
- **If building the planned tokenizer:** keep the custom BPE vocab size under roughly `12 × n_layer × n_embd` for whatever model size it's paired with — past that point, the tokenizer's output layer (`lm_head`) costs more compute per step than the rest of the transformer combined. For the `03-gpt` sizes tried so far (`n_embd=128, n_layer=6`), that threshold is ~9,200 tokens, so a custom vocab of a few hundred to a couple thousand should work fine.
- `torch.manual_seed(1337)` is fixed at the top of `attention.py` — fine for reproducibility, but it means every fresh run replays the same "random" batch sequence starting from iteration 0. 

## Results from attention.py training:
With `n_embd=128, n_layer=6, n_head=4, block_size=64` and 5000 iterations, validation loss is at ~1.51, example of produced output:
```
Second King Gentleman:
Why necorcase him from Christion greats,
Standing, true, to hear him discressor?
We must be professious arror half dry.

ROMEO:
You have spirit? O wise lord, art thou please.

POLIXENES:
My Servile, sir, what have them prove
And farmerlvess his richer vanting againsant: therefore
Within.
 
ISABELLAND LAUNIO:
Mercy, Grumio! of dying sovereign, and eating.
```

**Next steps:**
- Custom BPE tokenizer ([*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) / `minbpe`) to replace char-level tokenization in `03-gpt` — should fix most of the garbled-spelling issue above more directly than further scaling would.
- Implement checkpointing for gpt model, find larger dataset and try to achieve better loss.
