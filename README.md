# Grokking in a tiny transformer — the abstract flavour

This repository is the companion code for the article
**[Attention and grokking: a tiny transformer that learns the rule](https://jacano.github.io/blog/attention-and-grokking-tiny-transformer/)**.

It trains the **same model on the same task** as
[grokking-csharp](https://github.com/jacano/grokking-csharp), and it shows the same
jump. The difference is what is *not* here.

The micro flavour writes the engine: a list of nodes, a backward pass, the
derivatives of a matrix product, an optimizer by hand. It is 1,160 lines and it has
no dependencies. This flavour hands all of that to a framework and keeps only what
the reader actually thinks about: **what the model is and how it is trained**. The
model is 55 lines, and the training loop is 12.

| | micro flavour | abstract flavour |
| --- | ---: | ---: |
| model and training | 1,160 lines | **329 lines** |
| dependencies | none | TorchSharp + libtorch |
| backward pass | written by hand | `loss.backward()` |
| derivatives | written by hand | none |
| causal mask | the key cache grows | one argument |
| numbers | float64, own generator | float32, the framework's generator |
| a full run | about 10 minutes | **26 seconds** |
| the cliff | yes | yes |

## The library

**[TorchSharp](https://github.com/dotnet/TorchSharp)** 0.107, the .NET binding of
PyTorch, with the native build of libtorch 2.10 as the NuGet package
`libtorch-cpu`. There is no C++ compiler in the loop: the native library arrives as
a package.

It is the only serious candidate in .NET for this. `Microsoft.ML.TorchSharp` is a
wrapper for specific NLP scenarios and does not expose a general layer API;
`TensorFlow.NET` is a Keras-like binding with a heavier and less predictable native
dependency; and ML.NET has no autograd for a network you define yourself.

## The model, in full

This is the whole architecture. There is no other file that describes it.

```csharp
public override Tensor forward(Tensor index)
{
    long batch = index.shape[0];
    long length = index.shape[1];
    Tensor positions = arange(length, dtype: ScalarType.Int64, device: index.device);
    Tensor x = _wte.forward(index) + _wpe.forward(positions); // [B, T, C]

    // --- attention: the framework knows how to do this, and how to mask it ---
    Tensor h = RmsNorm(x);
    Tensor q = _wq.forward(h).reshape(batch, length, _heads, _headDim).transpose(1, 2);
    Tensor k = _wk.forward(h).reshape(batch, length, _heads, _headDim).transpose(1, 2);
    Tensor v = _wv.forward(h).reshape(batch, length, _heads, _headDim).transpose(1, 2);
    Tensor attended = scaled_dot_product_attention(q, k, v, is_casual: true);
    x = x + _wo.forward(attended.transpose(1, 2).reshape(batch, length, _embd));

    // --- MLP: project up, run the non-linearity, project back ---
    x = x + _fc2.forward(relu(_fc1.forward(RmsNorm(x))));
    return _lmHead.forward(x); // [B, T, vocab]
}
```

The three lines that replace the whole backward pass:

```csharp
Tensor loss = lossFn.forward(logits.reshape(-1, Vocab), targets.reshape(-1));
loss.backward();
optimiser.step();
```

And the two that replace the optimizer:

```csharp
using var optimiser = optim.Adam(model.parameters(), lr: lr, beta1: 0.85, beta2: 0.99, weight_decay: wd);
```

## The result

![Train and unseen accuracy against the training step. Train accuracy reaches 98% by step 1000 while unseen accuracy is at 1.2%. Unseen accuracy rises from 13% at step 3000 to 74% at step 3750 and reaches 97% by step 5000.](figures/grokking-cliff.svg)

| step | train acc | unseen acc | parameter size |
| ---: | ---: | ---: | ---: |
| 0 | 0.8% | 1.1% | 19.1 |
| 1,000 | 98.4% | 1.2% | 21.5 |
| 3,000 | 98.6% | 13.3% | 19.7 |
| 3,500 | 93.8% | 28.0% | 20.2 |
| 3,750 | 99.8% | 74.4% | 18.1 |
| 4,000 | 99.6% | 82.2% | 17.0 |
| 5,000 | 100% | 92.3% | 16.5 |
| 12,000 | 100% | **97.3%** | 16.0 |

The model memorizes the train set by step 1,000, and the unseen accuracy then sits
between 1% and 28% for another 2,500 steps. It jumps to 74% at step 3,750. At the
end the model answers **1,912 of the 1,966 unseen pairs**.

![Cross-entropy loss on a log scale. The train loss reaches its floor by step 1000 while the test loss stays flat at 3.9, then falls to 2.0 while the unseen accuracy jumps.](figures/grokking-loss.svg)

![Two curves against the training step, each on its own axis. On the left axis the size of the parameters rises from 19.1 to 21.5 while the model memorizes, then falls to 16.0. On the right axis the accuracy on unseen pairs stays near 1% and then rises to 97%.](figures/grokking-norm.svg)

The figure carries two axes, one per curve. The blue line is the size of the
parameters on the left axis. The red line is the accuracy on unseen pairs on the
right axis. The size tells the same story as in the micro flavour: it grows while
the model stores 843 answers, and it falls while the table becomes the expensive
option.

## The control

Set the decay to zero and the jump never arrives. The same program, with `--wd 0`:

| step | decay | train acc | unseen acc | size |
| ---: | :--- | ---: | ---: | ---: |
| 12,000 | with | 100% | **97.3%** | 16.0 |
| 12,000 | without | 100% | **0.2%** | 157.1 |
| 60,000 | without | 100% | **0.5%** | 348.2 |

```bash
dotnet run -c Release -- --wd 0
```

The model memorizes every training pair either way. Without the decay the unseen
accuracy is **below the 1.9% of guessing**, and it stays there: five times the
steps, 60,000 of them, move it from 0.2% to 0.5%. The size of the parameters climbs
to 348, more than twenty times the size of the model that learned the rule.

## Reproduce it

You need the .NET SDK 10 or newer. The first build downloads libtorch, which is a
few hundred megabytes.

```bash
git clone https://github.com/jacano/grokking-torchsharp
cd grokking-torchsharp
dotnet run -c Release
```

The whole run takes about half a minute. It writes the same three things as the
micro flavour:

| Where | What |
| --- | --- |
| `data/train.txt`, `data/test.txt` | the two splits of the dataset, one pair per line |
| `runs/grokking.csv` | the logged numbers of every step |
| `figures/` | the three figures of this article |

Three knobs, and `--eval-every`:

```bash
# control: no weight decay, so nothing pulls the model off the memorizing solution
dotnet run -c Release -- --wd 0

# a smaller modulus learns faster and shows the same shape
dotnet run -c Release -- --p 13 --steps 3000

# longer run, finer log
dotnet run -c Release -- --steps 40000 --eval-every 100
```

## Inference

`model.pt` is **not part of the repository**. The `--save` flag writes it at the end
of a run and the model reads it back. This is one line in each direction, because a
framework keeps the whole state dictionary for you:

```bash
dotnet run -c Release -- --save
dotnet run -c Release -- --infer 12+35
```

```
inference 12+35 = 47  [ok]  top: 47 (90%), 6 (5%), 17 (3%)
inference 7+44 = 51  [ok]  top: 51 (95%), 28 (2%), 39 (2%)
inference 50+50 = 5  [wrong]  top: 5 (74%), 17 (10%), 47 (7%)
inference 0+13 = 13  [ok]  top: 13 (96%), 1 (2%), 43 (0%)
```

One of the four is wrong, and that is the honest number: the model answers 97.3% of
the unseen pairs, so roughly one pair in forty fails. `50 + 50` is one of the 1,966
pairs the model never saw.

## What the framework does not do

The framework removes the arithmetic, not the decisions. You still choose the
architecture, the split, the number of steps, the learning rate and the weight
decay, and the run still has to be tuned. Two of the things this repository had to
get right:

- **The initialisation.** A framework starts an embedding at `N(0, 1)` and a linear
  layer in a uniform range. The article uses microgpt's `N(0, 0.08)`, so this file
  sets it in four lines. Measured by the size of the parameters, the two flavours
  then start at 19.1 and 19.2.
- **The decay is not the same decay.** `AdamW` subtracts the decay from the weight
  outside the update, and `Adam` adds it to the gradient, as microgpt does. With
  `AdamW` this run never jumped, at any decay from 0.002 to 0.5; the parameter norm
  settled at 40 to 50 and the model stayed on the memorizing answer. With `Adam` and
  the decay inside the gradient, **the same `wd = 0.0012` as the micro flavour
  works**, and the norm settles at 16. Two names for the same word, two different
  runs.

The framework also brings its own random numbers, so this model does not start from
the weights the micro flavour starts from, and it does not see the same split of the
pairs. Both reach the rule. The micro flavour is the one that shows how.
