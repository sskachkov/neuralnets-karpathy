# Neural Networks: Zero to Hero

Working through Andrej Karpathy's [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) — building neural nets from raw gradients up through a GPT-style transformer.

## Project layout
* [`01-micrograd/`](01-micrograd) From the backpropagation lecture. `Value`-based autograd engine (`+`, `*`, `**`, `tanh`, `exp`, topo-sort backward pass) + a small MLP (pure Python, no tensors). Gradients cross-checked against PyTorch's own autograd on the same computation.
* [`02-makemore/`](02-makemore) From makemore lectures, parts 1–3. Character-level name generator: bigram counts → single-layer NN → MLP w/ embeddings → deeper MLP w/ BatchNorm + init diagnostics.
* [`03-gpt/`](03-gpt) From the Let's build GPT lecture. Decoder-only transformer (multi-head self-attention, feedforward, residuals, layernorm), char-level, trained on Shakespeare dataset (shakespeare_complete_untagged.txt).
* [`04-tokenization`](04-tokenization) Followed Karpathy's lecture 'Let's build the GPT Tokenizer' to implement gpt2 like regex tokenizer.
* [`05-gpt_tokenized`](05-gpt_tokenized) Does not follow any lecture in particular. Reimplemented tokenizer with performance optimizations and ability to persist trained vocab and merges, the logic split into two classes BpeTrainer and BpeTokenizer.
Reimplemented gpt model, code is somewhat better organized and includes optimizations, like merged kqv vectors, pytorch built-in scaled_dot_product_attention and learning rate scheduler. Trained the model on shakespeare_complete.txt dataset.


## Running things

```sh
uv sync
source .venv/bin/activate
```
Then just run any `.py` file, for ex: `python3 03-gpt/attention.py` or `python3 05-gpt_tokenized/train_shakespeare.py`.
To open and run notebooks use a Jupyter-compatible editor.

`01-micrograd` requires native graphviz library, so `brew install graphviz` might be required.

Notebooks and scripts read from `datasets/` — input.txt and names.txt are from Karpathy repo, shakespeare_complete.txt and shakespeare_complete_untagged.txt are prepared from gutenberg project.

**Gotcha:** a `.py` script run from the repo root resolves `datasets/...` fine, but a notebook's kernel CWD defaults to *the notebook's own folder*, not the repo root — so from inside `02-makemore/`, the same path needs to be `../datasets/...`. 

## Notes 

- `01-micrograd` is pure Python/scalar — no tensors, no matmul, unlike the torch-based `02-makemore`/`03-gpt`. This is deliberate, so every step of the computation graph stays observable.
- **MPS can be *slower* than CPU** on `03-gpt` at small batch sizes (e.g. `batch_size=64`) — MPS has fixed per-op dispatch overhead that dominates when tensors are tiny. Bumping `batch_size` up (256 worked well) amortizes that overhead and flips MPS ahead of CPU. 
- **`03-gpt/attention.py` has no checkpointing.** A training run has to complete in one sitting — killing it partway loses everything. Worth remembering before kicking off a long run.
- **`03-gpt/attention.py`**  plateaus at ~1.50 loss; further tuning seems to yield little benefit. Next logical steps: a tokenizer and a larger training set (Upd: implemented in 05-gpt_tokenized).
- **`torch.manual_seed(31337)`** is fixed at the top of `attention.py` — fine for reproducibility, but it means every fresh run replays the same "random" batch sequence starting from iteration 0. 
- **`03-gpt/attention.py` vs `05-gpt_tokenized/train_shakespeare.py`** - after implementation of `05-gpt_tokenized`, I reran attention.py using the same shakespeare_complete dataset, as it's larger than what Karpathy uses, plus added bpc estimation into attention.py output, to make results comparable. There is still a difference in datasets, though: attention.py uses shakespeare_complete_untagged.txt, while train_shakespeare.py uses shakespeare_complete.txt, the difference is that latter has added '<|endoftext|>' text to separate scenes within plays.

## Results 
### Summary: char-level vs BPE
| | `03-gpt/attention.py` | `05-gpt_tokenized/train_shakespeare.py` |
|---|---|---|
| Tokenizer | char-level (93 symbols) | BPE, vocab 1024 (~2.36 chars/token) |
| Dataset | shakespeare_complete_untagged.txt | shakespeare_complete.txt |
| Params | 1.22M | 11.62M |
| Layers × n_embd × heads | 6 × 128 × 4 | 6 × 384 × 12 |
| Context | 64 chars | 512 tokens (~1200 chars) |
| Iterations × batch | 5000 × 256 | 8000 × 32 |
| Train time (CUDA) | 501 s | 3746 s |
| Final val loss (nats/token) | 1.4834 | 2.6737 |
| Final val bpc | 2.140 | 1.633 |

### 05-gpt_tokenized/train_shakespeare.py:
```
uv run 05-gpt_tokenized/train_shakespeare.py
loading cached tokenized dataset from cache/dataset_shakespeare_complete_bpe_tokenizer_vocab_1024_val_0.1_seed_31337.pt..
Model created, param count: 11.62M.
GPTTrainer created for dataset: shakespeare_complete, device: cuda, chars per token: 2.36, params: 11.62M,
config: ModelConfig(n_layer=6, batch_size=32, num_heads=12, n_embd=384, block_size=512, head_size=32, device='cuda', vocab_size=1024, special_tokens=['<|endoftext|>'], split_tokens=['<|endoftext|>'], dropout=0.3, lr=0.0005, val_ratio=0.1, seed=31337, verbose=True, dataset_name='shakespeare_complete', min_val_docs=20)
Starting training for 8000 iterations, train dataset size: 2.06M, val dataset size: 0.24M
Iter:    0, lr: 0.000005, train loss: 7.0056 (4.2791 bpc), val loss: 7.0062 (4.2795 bpc), elapsed: 20s (20 total).
Iter:  800, lr: 0.000493, train loss: 3.4379 (2.0999 bpc), val loss: 3.4800 (2.1256 bpc), elapsed: 371s (392 total).
Iter: 1600, lr: 0.000464, train loss: 2.8869 (1.7634 bpc), val loss: 2.9872 (1.8246 bpc), elapsed: 372s (764 total).
Iter: 2400, lr: 0.000415, train loss: 2.6743 (1.6335 bpc), val loss: 2.8535 (1.7430 bpc), elapsed: 372s (1137 total).
Iter: 3200, lr: 0.000353, train loss: 2.5514 (1.5584 bpc), val loss: 2.7831 (1.7000 bpc), elapsed: 372s (1509 total).
Iter: 4000, lr: 0.000282, train loss: 2.4647 (1.5055 bpc), val loss: 2.7288 (1.6668 bpc), elapsed: 372s (1882 total).
Iter: 4800, lr: 0.000211, train loss: 2.3815 (1.4547 bpc), val loss: 2.7020 (1.6505 bpc), elapsed: 372s (2255 total).
Iter: 5600, lr: 0.000146, train loss: 2.3209 (1.4176 bpc), val loss: 2.6900 (1.6431 bpc), elapsed: 372s (2628 total).
Iter: 6400, lr: 0.000095, train loss: 2.2774 (1.3911 bpc), val loss: 2.6876 (1.6416 bpc), elapsed: 372s (3001 total).
Iter: 7200, lr: 0.000061, train loss: 2.2469 (1.3724 bpc), val loss: 2.6638 (1.6271 bpc), elapsed: 372s (3373 total).
Iter: 7999, lr: 0.000050, train loss: 2.2288 (1.3614 bpc), val loss: 2.6737 (1.6331 bpc), elapsed: 372s (3746 total).
<|endoftext|>THE THIRD PART OF KING HENRY THE SIXTH
ACT V
SCENE III. London. A Room in the near Officers.

Enter the King, Lewis and Somerset.

LOUIS.
I am not afraid to speak to you 'tis dangerous.
For there's well may kindly victory defend
With that which you are attended.
My conscience can permit the work in war
Betwixt than fortune wing you bear forth well,
And,, like a watchless tricks of Saint Albans,
Must wear the napkin's partkers deemed,
And meet conquerors, the water-mell of your children
The spirit of mine own son uncle soul
Defile youth your noble father, in all steel?
Who's the Prince, countess, sir, that this gentleman,
Disdain intends his virtue in this full seas,
Whom I informed, arms, stout charity,
Than Clifford's countenance, and his envious title
Out of present wreathomenoms
Excepteding the mansion therein leaves him only,
But Warwick and Douglas, and Warwick that grows,
Incouncil, Duke of Somerset, that which the same
Was in the quarrel of York, whom of Lancaster,
The Duke of York force them he might;
And, since he was sent forthwith, he was late,
Knowing at the Duke of York, Duke of
Health the Duke of York, was there recorded,
From him that Northumberland or the Duke
Came of Rouen's, who, the painful Duke of York
Ireland's love mere, did feast for his hopeful face
Fick not before. The Duke of Gloucester's,
We were no stomach and laid upon the proposed,
Lord Havenspur and rebelsolute,
Yea, for our party supportable
Should equal with a revenue. The smallest stay the womb,
And who the multitude believed.

WARWICK.
The Duke of Norfolk, doth from Edwardhampton,
Mon him he should right the rude Queen.
The Duke of Suffolk, and commonwered him,
From whence the Duke already Henry,
Should poorly I exceed you, did crown him straight
To lay him reverence. Which I, Lord Protector,
And he was not yet unruly man, as he would,
To pray for all the justly wronged groans
And wishedom before you were, forsooth.

MORTIMER.
Be pleased, as myself, I should be armed
Or, by a brother, for mere overweened
That which wished serves, a tyrant of wars,
Not to be slain to have misbegotten used by a grave,
Only give us thinking words, and, pleasing powcious.
We have given this timeless growing of us,
And given him our truce. Hence, my liege,
I have no suit.

[Exit Mortimer.]

Now shall we repeal at supper and dignity,
After the gracious people's sides, Cardinal,
As for his majesty, give him comfort,
And with the stricting of the people.
```
### 03-gpt/attention.py:
With `n_embd=128, n_layer=6, n_head=4, block_size=64` and 5000 iterations, validation loss is at ~1.48, example of produced output:
```
uv run 03-gpt/attention.py
device: cuda
step 0: train loss 4.6764 (6.7466 bpc), val loss 4.6799 (6.7517 bpc)
step 500: train loss 1.7763 (2.5627 bpc), val loss 1.8295 (2.6393 bpc)
step 1000: train loss 1.5660 (2.2592 bpc), val loss 1.6648 (2.4018 bpc)
step 1500: train loss 1.4770 (2.1308 bpc), val loss 1.5942 (2.3000 bpc)
step 2000: train loss 1.4288 (2.0613 bpc), val loss 1.5593 (2.2496 bpc)
step 2500: train loss 1.4030 (2.0241 bpc), val loss 1.5370 (2.2174 bpc)
step 3000: train loss 1.3818 (1.9936 bpc), val loss 1.5216 (2.1952 bpc)
step 3500: train loss 1.3654 (1.9698 bpc), val loss 1.5129 (2.1826 bpc)
step 4000: train loss 1.3541 (1.9535 bpc), val loss 1.5036 (2.1692 bpc)
step 4500: train loss 1.3399 (1.9331 bpc), val loss 1.4895 (2.1489 bpc)
step 4999: train loss 1.3330 (1.9232 bpc), val loss 1.4834 (2.1401 bpc)
Training took: 501 seconds.
LORD.
By your sicks our lives, and masketh me so death.
All, o'er thou shalt not Queen,
They art buy honour in the horse,
And, nature that to talk out off itter.
And say this vile--
Alous paid exchose, Wullcassays,
She eft. Please me from alterty,
Unto kiss and and elj-star freeling the beasts.

ISABETELAN.
You have a given wit to once God,
My gild thou liest; when this fell was wore return that
In kin the?

GREMILVOLIO.
Smell, in the world,
The nimae high trippen,--have resembled of yours
swer his fits and accondets.

PROVOST;
That I were a wound, vilgard! Noll see the rear, yours, to spare a triumpeth this,
Free see and a coveror, and Nurse, amoant o'ers.

[Exit.]

DESDEMONA.
I commend thee, metor. In she satisfact'd,
So expresses, old them with a vouch
And worsh, in 't. [Good thrust rooked at foreign.]
 What, being;
'Twixt divont is penulies quake to the pain other did.
Yes, peace, none sinks dread me look then,
I'll tend me so nemies. Thou keep off change all.

Enter Oliver and Hyman's bark the Pulchase hand!
Welcondingman, Laer, blunt, lay! Nothing to enter my heart
Strights, better soon of all tellight, shall be
Down me for another word.

CALEON.
Who is? Why is not by me, I prithee not the wouch if I
For great a kind, in merchance service by it. No Nurs.

RICHARD.
Come, lady, till it not changed in love,--clight to abuse her.
I am worthine. A warlike and mark. Yet me
Sit of Acheries that had such as much, and then in my name and stomy,
Dungs of a voic'd that, of done.
Fareward the music say; away, as canst.

POGTH.
It is Kon appear:
To it is fash vars'd.

DUNT.
This Departing hail, stuffs, we is not so,
Promise quick him.

ARMER.
So keep you loves. How is ever, in very in terivity
And have all the paving army of the gods
That ladies, will ask been the readious are but male?

WHAMORATH.
Company, thou in footing, unkingdom I may beseech you urgen, no
fing, of one marved for it.

[Exeunt and takes after maids in thy warumper.]

DICK.
Graces gorge at most cheer ag
```

**Next steps:**
- Follow 'Let's reproduce GPT-2 (124M)' lecture and train a larger model using pytorch ddp.
