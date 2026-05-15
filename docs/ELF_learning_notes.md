# ELF Learning Notes

A probe-style walkthrough of the ELF paper (Embedded Language Flows, paper-2605.10938).
Captured as Q -> A so the chain of reasoning is preserved, not just the conclusions.

---

## 1. What is ELF in one breath?

Continuous flow-matching model for language.
Denoises in continuous embedding space at every step except the last.
The last step is the only place that discrete tokens appear.

```python
noise_embedding -> ... denoise ... -> clean_embedding -> tokens
```

No separate decoder. The same network weights do both denoising and final decoding,
switched by a "mode" flag.

---

## 2. Core training pseudocode

```python
def train_step(tokens):
    x = encode(tokens)                         # T5 encoder embeddings

    if random() < 0.8:
        # ---- denoise branch ----
        t = sample_t()                         # logit-normal in [0,1]
        noise = randn_like(x)
        z_t = t * x + (1 - t) * noise          # rectified flow path

        x_pred = net(z_t, t, mode="denoise")

        v_pred = (x_pred - z_t) / (1 - t)
        v_true = (x      - z_t) / (1 - t)
        loss = mse(v_pred, v_true)             # == mse(x_pred, x) / (1-t)^2
    else:
        # ---- decode branch (t = 1) ----
        z = corrupt(x)                         # non-trivial near-clean input
        x_pred, logits = net(z, t=1, mode="decode")
        loss = cross_entropy(logits, tokens)

    update(loss)
```

Paper default mix: 80% denoise, 20% decode.

---

## 3. Why velocity loss instead of plain `mse(x_pred, x)`?

It is the same target reweighted by time:

```python
mse(v_pred, v_true) == mse(x_pred, x) / (1 - t)**2
```

So it is just a time-dependent loss weight:

```python
t = 0.1  -> weight ~ 1.23
t = 0.5  -> weight = 4
t = 0.9  -> weight = 100
t = 0.99 -> weight = 10000
```

Reason: the ODE sampler divides by `(1 - t)`, so errors near `t=1` blow up
into large velocity errors. The reweighting matches sampler sensitivity.

Skeptical take: this is just "weight harder/more-sensitive steps more".
The "deeper" idea is not the `1/(1-t)^2` factor — that is bookkeeping.
The deeper idea is the next section.

---

## 4. The real bet of ELF

```python
# Older continuous DLMs: discretize at every step.
for t in schedule:
    logits = decode(net(z_t, t))
    loss += cross_entropy(logits, tokens)

# ELF: stay continuous, discretize only at the end.
for t in schedule[:-1]:
    loss += flow_loss(net(z_t, t), x)
loss += cross_entropy(net(z_near_clean, t=1, mode="decode"), tokens)
```

So: let the diffusion live in continuous space; force tokens only at the final step.
The unembedding matrix is a linear projection sharing weights with the denoiser.

---

## 5. The "two modes" inside the net

Not two networks. One transformer, conditioned on a `mode` flag.

```python
mode = "denoise"  ->  mode_tokens zeroed,    no decoder head,  output = x_pred
mode = "decode"   ->  mode_tokens active,    decoder head on,  output = x_pred + logits
```

Implementation detail (from `modules/model.py`):

```python
# prefix structure inside the transformer:
[ time_tokens ] [ mode_tokens ] [ actual sequence tokens ]
```

- `time_tokens`: learned prefix + sinusoidal time embedding (in-context conditioning,
  cheaper than AdaLN-Zero).
- `mode_tokens`: learned tokens, multiplied by 0 in denoise mode, by 1 in decode mode.
- decoder head: `gelu(h @ proj) @ unembed -> [B, L, vocab]`, only active in decode mode.

Block structure is standard DiT-flavoured transformer:

```python
def ELFBlock(h):
    h = h + attention(rms_norm(h))   # qk-norm, RoPE
    h = h + swiglu_mlp(rms_norm(h))
    return h
```

---

## 6. "Depth" here

```python
depth = number_of_transformer_blocks

ELF-T  = 4
ELF-XS = 6
ELF-B  = 12
ELF-M  = 24
ELF-L  = 32
```

Our laptop TinyStories run used `ELF-T` (depth=4, hidden=128, heads=4).

Not to be confused with:

```python
sequence_length  # token positions
hidden_size      # transformer width
depth            # transformer blocks
sampling_steps   # denoising iterations
```

---

## 7. How is it different from GPT?

The block is not the surprise. It is a bidirectional transformer, BERT/DiT-style.

The real differences are:

```python
# GPT
p(s1..sL) = prod p(s_i | s_<i)
generate: append one token at a time

# ELF
z = randn([B, L, D])
for t in schedule:
    z = denoise(z, t)
tokens = decode(z)
```

- attention is bidirectional during denoising
- generates all positions at once (parallel decoding)
- can revise earlier positions
- inference cost is O(denoise_steps) sequential, not O(sequence_length) sequential

---

## 8. Length handling

There is a hard max:

```python
max_length  # fixed canvas length

# our TinyStories run:
max_length = 128

# paper-scale OWT run:
max_length = 1024
```

Variable length is handled by EOS/PAD inside the canvas:

```python
"Once upon a time ... the end. <eos> <pad> <pad> ..."
```

So:

```python
length < max_length   -> natural, EOS/pad fills the rest
length == max_length  -> natural
length > max_length   -> NOT natural;
                          need chunking / sliding window / hierarchical generation
```

ELF is not magically long-memory. Same window limit problem as GPT.

---

## 9. Why pursue this at all?

Not because of memory. Because of:

1. Fewer sequential steps:
   ```python
   GPT: ~L sequential forward passes
   ELF: ~K sequential forward passes, K << L
   ```
2. Bidirectional global planning per step.
3. Refinement / revision of earlier tokens.
4. Natural fit for infilling, editing, rewriting, translation, summarization.
5. CFG (classifier-free guidance) is natural in continuous space; weak in discrete DLMs.
6. Paper shows: lower generative perplexity, fewer sampling steps, ~10x fewer
   training tokens, no distillation needed — vs leading discrete DLMs and prior
   continuous DLMs.

Weakness: open-ended unbounded continuation is still more natural in GPT.

---

## 10. Full inference recap

```python
z = randn([B, L, D])

for t, next_t in time_schedule:                    # e.g. 16/32/64 steps
    x_pred, _ = net(z, t, mode="denoise")
    v = (x_pred - z) / (1 - t)
    z = z + (next_t - t) * v                       # Euler step

_, logits = net(z, t=1, mode="decode")             # one decode pass
tokens = argmax(logits)
```

Two flavors of sampler:

- **ODE**: deterministic Euler step above.
- **SDE-inspired**: inject small noise per step + shift `t` back toward noise regime
  (paper Alg. for the sampler in appendix).

---

## 11. Add-ons on top of the core

- **Self-conditioning**: feed previous `x_pred` back into the next forward pass.
  Implemented by concatenating `[z_t, x_pred_prev]` as the network input.
- **Classifier-Free Guidance (CFG)**: combine conditional and unconditional
  predictions. ELF uses training-time CFG so inference does not need two passes.
- **Conditional generation**: prepend clean condition embeddings (e.g. prompt /
  source sentence) to the input; they are never corrupted; the model attends to
  them via self-attention.

---

## 13. When is ELF actually worth it? (my own conclusion)

Best fit: tasks where the output length is roughly fixed and non-trivially long.

```python
good_fit = [
    "translation",
    "summarization",
    "code completion of known scope",
    "structured outputs (JSON, tables)",
    "infilling / editing",
    "constrained rewriting",
]

bad_fit = [
    "open chat",
    "long-form essays where the length IS the answer",
    "very short outputs (canvas wasted on padding)",
]
```

### Sharper version of the inference cost story

The right quantity to compare is *sequential depth*, not raw FLOPs.

```python
# Latency / sequential depth
GPT  = L  forward passes (one per token, KV-cached, each cheap)
ELF  = K  forward passes (each is over the full sequence)

# Raw FLOPs (roughly)
GPT  ~ L^2       # KV cache => each new token O(L)
ELF  ~ K * L^2   # each step is full attention over L

# Ratio
elf_flops / gpt_flops ~ K
```

So ELF trades MORE total compute for FEWER sequential steps.

Concrete example:

```python
L = 1000   # output length
K = 32     # denoising steps

GPT_latency = 1000 sequential steps
ELF_latency =   32 sequential steps   # ~30x fewer

GPT_flops   ~  1_000_000
ELF_flops   ~ 32_000_000              # ~32x more
```

ELF wins when:

```python
you_have = "lots of parallel compute, latency matters"
```

ELF loses when:

```python
you_have = "tight FLOPs budget, one stream, no parallelism to spend"
```

This is also why diffusion-style methods are appealing for serving:
a GPU is rarely the bottleneck for one user; latency is.

### The "thinking time" knob

This is one of the strongest reasons to care.

```python
# GPT thinking-time scaling = chain-of-thought, more output tokens
# ELF  thinking-time scaling = more denoising steps on the SAME output length

K = 8     # fast, lower quality
K = 32    # default
K = 128   # slower, refined
K = 256   # very refined
```

So you get a dial that does not change the output length, only the quality:

```python
output_length  = fixed
thinking_time  = K (denoising steps)
quality        = increases with K (diminishing returns)
```

GPT does not have a clean version of this. CoT mixes "thinking" with "output".
ELF cleanly separates them.

### Floor caveat

Even short-but-not-tiny is not enough. There is a lower bound where
GPT with KV cache is simply faster.

```python
ELF_attractive when:
    L >> K
    AND L is large enough that L sequential steps actually hurt latency

ELF_not_attractive when:
    L is small (GPT finishes in a few cached steps anyway)
    OR L is mostly padding (we paid K*L^2 for a 50-token answer)
```

### TL;DR

```python
ELF really shines when:
    output_length >> denoising_steps
    output_length is mostly used (not 90% pad)
    latency matters more than raw FLOPs
    optional refinement / "thinking time" is a feature
```

---

## 12. One-paragraph mental model

ELF treats a sentence as a sequence of points in T5 embedding space. Training
shows the network many noisy versions of these point-clouds at different noise
levels, and asks it to predict the clean cloud (with a velocity-style loss
weighted toward the cleaner end of the path). On rare batches it also asks the
same network, in "decode" mode, to map a near-clean cloud back to discrete
tokens via a linear unembedding. At inference, we start from pure Gaussian noise
in this embedding space, walk it toward clean embeddings with a handful of ODE
or SDE steps, and discretize once at the end. The result: text generation that
is parallel across positions, bidirectional, revisable, and cheap in
sequential-step count compared to autoregressive models.
