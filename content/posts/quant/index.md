---
title: "Quantization, from one block of weights to 4-bit training"
date: 2026-11-15
draft: false
tags: ["LLM", "quantization", "inference", "FP4"]
summary: "What a quantizer does and what it speeds up, how to read E2M1, E4M3 and E8M0, block vs tensor scales, MXFP4 vs NVFP4 worked by hand, GPTQ and AWQ with runnable code, the GGUF format, and how DeepSeek trained on an FP4 grid."
ShowToc: true
TocOpen: false
---

An 8B model in BF16 takes 16 GB. In 4 bits it takes about 4.5–5.5 GB: 8B × 0.5 byte = 4 GB is the floor, the block scales add ~0.25–0.5 bit per weight, and the embedding and output head (~0.5B parameters each in a Llama-3-8B-class model) usually stay in BF16 (a Q4_K_M GGUF of Llama-3-8B is ~4.9 GB, an AWQ checkpoint with an FP16 head ~5.7 GB). Quantization is the art of making that cut without the model noticing. This post goes through the pieces in the order they build on each other: what a quantizer does, what gets faster, the number formats (INT4, FP8, FP4 and their scales), the two classic post-training methods (GPTQ and AWQ), the GGUF file format, and how DeepSeek trained models that live on a 4-bit grid.

## What gets faster

Decoding one token reads every weight once and does very little arithmetic per byte, so it is limited by memory bandwidth (see the KV cache post: 16 GB / 4.8 TB/s = 3.3 ms per step on an H200). Prefill and training are limited by compute instead. That splits quantization into two families:

| Scheme | What is low precision | What it speeds up |
|---|---|---|
| **W4A16** (GPTQ, AWQ, GGUF) | Weights stored in 4 bits; kernels expand them to BF16 on the fly | Decoding: a quarter of the bytes to read, so up to ~4× faster. Prefill barely changes: the math still runs in BF16 |
| **W8A8 / FP8** | Weights and activations, so the matmul itself runs on FP8 tensor cores | Compute-bound work too: prefill, large batches, training (DeepSeek-V3) |
| **W4A4 (NVFP4, MXFP4)** | Both sides in 4 bits, on Blackwell FP4 tensor cores | Both: half the bytes of FP8 and twice the FP8 math rate |

### Loading an FP4 model on an H200

An H200 (Hopper) has no FP4 tensor cores, yet DeepSeek-V4-Flash, with FP4 experts, loads and runs on it. Loading only needs memory, and memory holds any bits: an FP4 checkpoint packs two 4-bit codes into each byte (stored as plain `uint8`), with the scales in a separate tensor. What Hopper can't do is arithmetic on FP4, so every matmul kernel expands the weights on chip, one tile at a time:

```text
HBM:          FP4 codes (0.5 byte each) + scales          ← what stays in memory
  ↓ load one tile into registers / shared memory
on chip:      w = code × scale  → FP8 or BF16              ← expanded here, a tile at a time
  ↓
tensor cores: FP8 / BF16 matmul
```

The full weight is never written back to memory in FP8 or BF16; that would undo the saving. So on an H200 FP4 still buys **memory** (a 284B model with FP4 routed experts fits on one 8×H200 node) and **fewer bytes read** (faster decode), but not faster arithmetic. On Blackwell, the tensor cores take FP4 codes and scales directly.

Training does the same expansion in the forward pass. The weights being **updated**, though, can't be FP4: a gradient step is far smaller than the gap between FP4 grid points and would round away. Training keeps full-precision weights somewhere: DeepSeek's QAT keeps FP32 master weights (see PTQ vs QAT below), and a LoRA run keeps the base frozen and trains a BF16 adapter.

## The basic recipe: scale, round, store

Every scheme in this post does the same three steps on a small group of weights:

```text
scale = max |w| in the group / largest grid value      (INT4: 7, FP4: 6)
q     = round(w / scale) to the nearest grid value      (stored in 4 bits)
w'    = q × scale                                       (what the matmul sees)
```

The scale is stored alongside, so the real cost per weight is the element bits plus the scale's share:

| Layout | Bits per weight |
|---|---|
| INT4, FP16 scale per 128 values | 4 + 16/128 = **4.125** |
| MXFP4: FP4, 8-bit scale per 32 | 4 + 8/32 = **4.25** |
| NVFP4: FP4, 8-bit scale per 16 (+ one FP32 per tensor) | 4 + 8/16 = **4.5** |

**Outliers are the whole problem.** The group's scale is set by its largest value. One weight that is 10× bigger than its neighbours stretches the step size for the whole group, and the small weights all round to zero. Smaller groups (per tensor → per channel → 128 → 32 → 16) confine the damage to fewer neighbours; the price is more scales to store.

## Number formats

### Reading a name like E4M3

E = exponent bits, M = mantissa bits, plus one sign bit. Every binary float stores ±1.xxx × 2^exponent:

```text
normal (E ≠ 0):     value = ±2^(E − bias) × (1 + M / 2^m)
subnormal (E = 0):  value = ±2^(1 − bias) × (M / 2^m)
bias = 2^(k−1) − 1 for k exponent bits:   k=2 → 1,  k=4 → 7,  k=8 → 127
```

The leading 1 is implicit (not stored), the bias lets an unsigned field hold negative exponents, and subnormals fill the gap down to zero.

### FP4 (E2M1): only 8 magnitudes

Two exponent bits and one mantissa bit give exactly ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}:

```text
E   M   kind        calculation             value
00  0   subnormal   2^0 × 0/2               0
00  1   subnormal   2^0 × 1/2               0.5
01  0   normal      2^0 × (1 + 0/2)         1
01  1   normal      2^0 × (1 + 1/2)         1.5
10  0   normal      2^1 × 1                 2
10  1   normal      2^1 × 1.5               3
11  0   normal      2^2 × 1                 4
11  1   normal      2^2 × 1.5               6
```

No infinity, no NaN. Compared with INT4 (−8…7, evenly spaced), FP4 puts more points near zero, where most weights are, and fewer near the maximum. Everything else comes from the shared scale, which is why the scale format matters so much at 4 bits.

### What the scale does

The scale turns the 8 grid magnitudes into real weight values. Divide to quantize, multiply to use:

```text
quantize:   code = round(w / s)       w = real weight, s = block scale
use:        w'   = code × s           what the tensor cores see (FP8 / BF16)
```

A block whose largest weight is 0.031 gets s = 2⁻⁷ = 0.0078 under MXFP4 (how, in the next subsection). Its possible values are the grid times s:

```text
codes:  0     0.5     1       1.5     2       3       4       6
× s:    0     0.0039  0.0078  0.0117  0.0156  0.0234  0.0312  0.0469
```

The result is an ordinary FP8 or BF16 number; "4-bit" means each weight is stored as one of 16 codes. Inside one block there are still only 16 possible values. Finer values come from different blocks having different scales: a block of tiny weights gets s = 2⁻¹², a block of large ones s = 2⁻⁵, so each gets a grid sized for it. That is why block size matters.

### FP8 (E4M3): max 448

```text
value = ±2^(E − 7) × (1 + M/8)        8 values per power of two, step ≈ 12.5%
0 1111 110   2^8 × 1.75   = 448       (maximum)
0 0001 000   2^−6         = 0.0156    (smallest normal)
0 0000 001   2^−6 × 1/8   = 2^−9      (smallest subnormal)
```

The "fn" variant used in ML has no infinity and only one NaN code, so the top exponent holds real numbers: the maximum is 448 rather than the IEEE-style 240. The range is narrow, about 17–18 powers of two. That is fine for a value relative to a tensor maximum, not for raw weights.

### E8M0: a scale that is just an exponent

Eight exponent bits, no mantissa, no sign: value = 2^(E − 127). It is exactly the exponent byte of an FP32 number, so multiplying by it is an exponent add, and it covers any magnitude. The catch: only powers of two. A block that needs a scale of 1.3 gets 2, so its largest value lands somewhere in (3, 6] on the FP4 grid instead of near 6, wasting up to one bit.

### Smaller blocks or a finer scale?

Halving the block to 8 costs 4 + 8/8 = 5 bits per weight. It lowers the error, but with an E8M0 scale much less than you'd expect, because the power-of-two rounding of the scale costs more than the block size does. Measured on random weights (relative error):

| Scale, block | Bits / weight | Gaussian | Heavy-tailed (t₃) |
|---|---|---|---|
| E8M0, 32 (MXFP4) | 4.25 | 11.5% | 15.1% |
| E8M0, 16 | 4.5 | 11.3% | 13.3% |
| E8M0, 8 | 5.0 | 11.0% | 12.0% |
| **E4M3, 16 (NVFP4)** | **4.5** | **9.5%** | **9.0%** |
| E4M3, 8 | 5.0 | 8.5% | 7.5% |

NVFP4 at 4.5 bits beats E8M0 blocks of 8 at 5 bits: a finer scale buys more than a smaller block. Hardware also limits the choice: FP4 tensor cores take blocks of 32 (MX) and 16 (NVFP4) only, so block 8 needs a custom kernel.

### Picking an MXFP4 scale

Use the smallest power of two that keeps the block's largest value within FP4's maximum of 6:

```text
s = 2^ceil(log2(amax / 6))

amax = 0.031:   0.031 / 6 = 0.005167,  log2 = −7.60,  ceil → −7,  s = 2⁻⁷ = 0.0078
                check: 0.031 / 0.0078 = 3.97 ≤ 6 ✓
                round down instead: s = 2⁻⁸ → 0.031 / 0.0039 = 7.94 > 6, the max would clip
```

Two more blocks:

```text
block [0.9, −0.3, 0.05, 0.6]
amax / 6 = 0.15,  log2 = −2.74,  ceil → −2,  s = 0.25
grid × s: 0  0.125  0.25  0.375  0.5  0.75  1.0  1.5
w / s   = [3.6, −1.2, 0.2, 2.4]   →  round [4, −1, 0, 2]   →  w' = [1.0, −0.25, 0, 0.5]
error   = 11%, 17%, 100%, 17%      (0.05 is too small for this grid and becomes 0)

block [200, 50, −120, 7]
amax / 6 = 33.3,  log2 = 5.06,  ceil → 6,  s = 64
w / s   = [3.125, 0.78, −1.875, 0.11]   →  round [3, 1, −2, 0]   →  w' = [192, 64, −128, 0]
```

These show the cost of a power-of-two scale: the largest value lands at 3.97, 3.6 or 3.1 instead of near 6, so the top of the grid goes unused. NVFP4's finer E4M3 scale closes that gap.

One caveat: the OCP MX spec defines the scale as `2^(floor(log2(amax)) − 2)`, where 2 is E2M1's largest exponent. For some blocks that is one step smaller and the largest value is clipped to 6 (for 0.031 it gives 2⁻⁸, so 7.94 clips to 6). The code below uses the round-up rule, which never clips; real implementations differ.

### Block scale vs tensor scale

A **block scale** is one number per small group (32 values for MXFP4, 16 for NVFP4, a 128×128 tile for DeepSeek-V3's FP8 weights). A **tensor scale** is one FP32 number for the whole matrix. NVFP4 needs both: its block scale is E4M3, which only spans ~17 powers of two, and raw weights (around 10⁻³) sit outside that. Divide by the tensor scale first and the block scales land inside E4M3's range. MXFP4's E8M0 covers everything on its own, so it has no tensor scale.

```text
effective scale = tensor scale (FP32) × block scale (E4M3)       NVFP4
effective scale = block scale (E8M0)                             MXFP4
```

## MXFP4 vs NVFP4, one block by hand

Take four weights [0.031, −0.012, 0.0047, 0.020] as one block (a real block has 32 or 16):

```text
MXFP4
scale  = 2^ceil(log2(0.031/6)) = 2^−7 = 0.0078        (E8M0, power of two)
w/s    = [3.97, −1.54, 0.60, 2.56]
round  = [4, −1.5, 0.5, 3]
back   = [0.0312, −0.0117, 0.0039, 0.0234]
error  = 0.8%, 2.3%, 17%, 17%

NVFP4  (tensor max 0.25 → tensor scale g = 0.25/(6·448) = 9.3e−5)
block  = 0.031/6 / g = 55.6 → E4M3 rounds to 56;  scale = 56·g = 0.00521
w/s    = [5.95, −2.30, 0.90, 3.84]
round  = [6, −2, 1, 4]
back   = [0.0312, −0.0104, 0.0052, 0.0208]
error  = 0.8%, 13%, 11%, 4%
```

The error energy is about 3.5× lower for NVFP4 here. The power-of-two scale made MXFP4 put the block's largest value at 3.97 instead of near 6, so it used less of the grid. Individual weights can go either way (the second one is better in MXFP4).

| | MXFP4 | NVFP4 |
|---|---|---|
| Element | E2M1 | E2M1 |
| Block | 32 | 16 |
| Block scale | E8M0 (power of two) | E4M3 (3 mantissa bits) |
| Tensor scale | none | FP32 |
| Bits per weight | 4.25 | 4.5 |
| Standard | OCP Microscaling, multi-vendor | NVIDIA |
| Native FP4 math | Blackwell, AMD MI355 | Blackwell |
| Pros | Simple, no calibration; power-of-two scales fold exactly into FP8 | More accurate, especially with heavy tails; built for 4-bit activations too |
| Cons | Wastes up to a bit per block; 32-value blocks feel outliers more | ~6% more memory; needs a per-tensor max; NVIDIA-only |

## Code: the quantizers from scratch

Everything above, plus toy versions of GPTQ and AWQ (next sections), in about 90 lines of PyTorch. It runs on a CPU in a few seconds:

```python
"""Quantizers from scratch: INT4 groups, MXFP4, NVFP4, plus toy GPTQ and AWQ. CPU only."""
import torch
torch.manual_seed(0)

FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])           # E2M1 magnitudes
FP4 = torch.cat([-FP4.flip(0), FP4])

def round_to_grid(x, grid):                                # nearest grid value
    return grid[(x[..., None] - grid).abs().argmin(-1)]

def e4m3(x):                                               # round positive x to FP8 E4M3 (max 448)
    x = x.clamp(max=448.0)
    e = torch.floor(torch.log2(x.clamp(min=2**-9))).clamp(min=-6)   # below 2^-6: subnormal step 2^-9
    step = 2.0 ** (e - 3)                                  # 8 values per power of two
    return torch.round(x / step) * step

# ---- three block quantizers; w is [rows, cols], blocks run along a row ----
def int4_group(w, g=128):                                  # symmetric INT4, FP16 scale per g values
    b = w.reshape(-1, g)
    s = b.abs().amax(1, keepdim=True) / 7                  # map the group max to 7
    q = torch.round(b / s).clamp(-8, 7)
    return (q * s).reshape(w.shape)

def mxfp4(w, g=32):                                        # E2M1 values, E8M0 (power-of-two) scale per 32
    b = w.reshape(-1, g)
    s = 2.0 ** torch.ceil(torch.log2(b.abs().amax(1, keepdim=True) / 6))
    return (round_to_grid(b / s, FP4) * s).reshape(w.shape)

def nvfp4(w, g=16):                                        # E2M1 values, E4M3 scale per 16, FP32 per tensor
    t = w.abs().max() / (6 * 448)                          # tensor scale
    b = w.reshape(-1, g)
    s = e4m3(b.abs().amax(1, keepdim=True) / 6 / t) * t    # block scale, stored in FP8
    return (round_to_grid(b / s, FP4) * s).reshape(w.shape)

def rel_err(w, wq): return ((w - wq).norm() / w.norm()).item()

# 1) one block, by hand
blk = torch.tensor([[0.031, -0.012, 0.0047, 0.020]])
print("block       ", [round(v, 4) for v in blk.tolist()[0]])
print("MXFP4       ", [round(v, 4) for v in mxfp4(blk, 4).tolist()[0]])
t = 0.25 / (6 * 448); s = e4m3(torch.tensor(0.031 / 6 / t)) * t   # pretend the tensor max is 0.25
print("NVFP4       ", [round(v, 4) for v in (round_to_grid(blk / s, FP4) * s).tolist()[0]])

# 2) error on a whole matrix
for name, w in [("gaussian", torch.randn(4096, 4096) * 0.02),
                ("heavy-tailed", torch.distributions.StudentT(3).sample((4096, 4096)) * 0.02)]:
    print(f"{name:13s} INT4/g128 {rel_err(w, int4_group(w)):.1%}   MXFP4 {rel_err(w, mxfp4(w)):.1%}   NVFP4 {rel_err(w, nvfp4(w)):.1%}")

# 3) GPTQ vs round-to-nearest on one linear layer (INT4, g=128)
d_in, d_out, n = 512, 256, 2048
X = torch.randn(n, d_in) @ torch.randn(d_in, d_in) * 0.1    # correlated calibration inputs
W = torch.randn(d_out, d_in) * 0.02

def gptq(W, X, g=128):
    W = W.clone(); Q = torch.zeros_like(W)
    H = X.T @ X; H += 0.01 * H.diag().mean() * torch.eye(len(H))       # damping
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    for j in range(W.shape[1]):
        if j % g == 0:                                     # scale for the next group, from current weights
            s = W[:, j:j + g].abs().amax(1) / 7
        Q[:, j] = torch.round(W[:, j] / s).clamp(-8, 7) * s
        err = (W[:, j] - Q[:, j]) / Hinv[j, j]
        W[:, j + 1:] -= err[:, None] * Hinv[j, j + 1:]    # push the error onto later columns
    return Q

def out_err(Wq): return ((X @ W.T - X @ Wq.T).norm() / (X @ W.T).norm()).item()
print(f"layer output error   RTN {out_err(int4_group(W)):.2%}   GPTQ {out_err(gptq(W, X)):.2%}")

# 4) AWQ: scale up the channels whose inputs are large
X2 = torch.randn(n, d_in); X2[:, :5] *= 20                 # 5 channels carry large activations
def awq(W, X, g=128):
    a = X.abs().mean(0)                                    # per-input-channel activation size
    best = None
    for alpha in torch.linspace(0, 1, 21):                 # grid search, as in the paper
        s = a ** alpha; s = s / (s.max() * s.min()).sqrt()         # normalise around 1
        Wq = int4_group(W * s, g) / s                      # W·diag(s), then X·diag(1/s) folded back
        e = (X @ W.T - X @ Wq.T).norm()
        if best is None or e < best[0]: best = (e, Wq, alpha.item())
    return best[1], best[2]
def out_err2(Wq): return ((X2 @ W.T - X2 @ Wq.T).norm() / (X2 @ W.T).norm()).item()
Wq, alpha = awq(W, X2)
print(f"layer output error   RTN {out_err2(int4_group(W)):.2%}   AWQ {out_err2(Wq):.2%}  (alpha={alpha:.2f})")
```

Output:

```text
block        [0.031, -0.012, 0.0047, 0.02]
MXFP4        [0.0312, -0.0117, 0.0039, 0.0234]
NVFP4        [0.0312, -0.0104, 0.0052, 0.0208]
gaussian      INT4/g128 11.7%   MXFP4 11.9%   NVFP4 9.5%
heavy-tailed  INT4/g128 22.4%   MXFP4 15.0%   NVFP4 9.0%
layer output error   RTN 11.71%   GPTQ 8.07%
layer output error   RTN 12.13%   AWQ 8.17%  (alpha=0.25)
```

What to read from it:

- The block matches the hand calculation.
- On Gaussian weights the three 4-bit schemes are close (~10–12% error). On heavy-tailed weights (Student-t, a stand-in for real LLM weights with outliers), INT4 with 128-value groups doubles its error, MXFP4 suffers a bit, and NVFP4 holds: small blocks and fine scales are what handle outliers.
- GPTQ and AWQ each cut the **layer output** error by about a third compared with plain rounding (RTN), with the same 4-bit format. They don't change the format; they change which weights get rounded which way.

## PTQ vs QAT

- **Post-training quantization (PTQ)** converts a finished model, using a few hundred calibration samples and no training. GPTQ, AWQ and GGUF's k-quants are all PTQ.
- **Quantization-aware training (QAT)** runs the forward pass on quantized weights during training, so the model learns to live on the grid. Gradients pass straight through the rounding to full-precision master weights (the straight-through estimator). DeepSeek-V4 does this for its experts.

PTQ is cheap and works well at 8 bits and usually at 4. QAT costs a training run but is what makes 4 bits (and lower) safe.

### One QAT step: quantize, dequantize, update

QAT is quantize-dequantize, also called fake quantization or QDQ:

```text
forward:   w (FP32)  ─quantize→  code = round(w / s), s  ─dequantize→  w' = code × s  →  matmul
backward:  treat round() as identity:  ∂L/∂w = ∂L/∂w'        (straight-through estimator)
update:    w ← w − lr · (Adam step), in FP32
```

The forward pass sees the rounded weights, so the loss measures the model on the real deployment grid. The gradient skips the rounding (round() has zero gradient almost everywhere). The update lands in the FP32 master weight, where small changes add up until the weight crosses a rounding boundary and its FP4 code moves to the next grid value. When training ends, only the codes and scales are kept: that is the released FP4 checkpoint.

### The straight-through estimator (STE)

Q is a staircase: flat between jumps, so its true slope is 0 almost everywhere (and undefined at the jumps).

```text
Ŵ = Q(W)
 6 |            ┌───
 4 |        ┌───┘
 3 |    ┌───┘
 2 |────┘
   └──────────────────  W
true dQ/dW = 0  →  ∂L/∂W = ∂L/∂Ŵ · 0 = 0  →  W never changes
```

The STE replaces that slope with 1 in the backward pass, so the gradient passes "straight through" the rounding unchanged:

```text
forward:   use Q(W)                  (the real rounded value)
backward:  ∂L/∂W := ∂L/∂Ŵ            (as if Ŵ = W)
```

In PyTorch it is one line:

```python
w_hat = w + (quantize(w) - w).detach()   # value = quantize(w); gradient w.r.t. w = 1
```

The gradient is "wrong" (a biased estimate), but it points the right way: if the loss wants Ŵ larger, it nudges W larger, and eventually W crosses a step and Ŵ really does get larger.

**Clipped weights get no gradient.** FP4's largest value is 6, so anything beyond 6·s is clipped to ±6·s. There Q is flat forever: with s = 1, W = 7.3, 8, 9 or 20 all give Q(W) = 6. A plain STE would keep passing "make it bigger", W would drift without bound, and the forward pass would never change. So the gradient is masked:

```text
∂L/∂W = ∂L/∂Ŵ   if |W/s| ≤ 6       inside the grid: straight through
      = 0        if |W/s| > 6       saturated: changing W cannot change the forward
```

With the round-up MXFP4 scale above nothing ever clips. Clipping does happen with the OCP spec's scale rule, when NVFP4's E4M3 block scale rounds down, with INT formats (clamped to −8…7), and with fixed or learned scales (e.g. LSQ).

### QAT for a LoRA

To merge a LoRA into an FP4 base without loss, put the quantizer inside the adapter's forward pass:

```text
Ŵ = Q(W₀ + (α/r)·B·A),     G = ∂L/∂Ŵ   (through the STE)
∂L/∂B = (α/r) · G · Aᵀ,    ∂L/∂A = (α/r) · Bᵀ · G
```

Where L comes from, for an SFT run, end to end:

```text
forward, every adapted layer:   Ŵ = Q(W₀ + (α/r)·B·A),    y = Ŵ x        (W₀ frozen; only A, B train)
loss, next-token cross-entropy on the response tokens only (prompt tokens masked out):
    z_t = logits at position t,    p_t = softmax(z_t)
    L   = −(1/N) Σ_{t ∈ response} log p_t[y_t]               N = number of response tokens
backward:
    ∂L/∂z_t = (p_t − onehot(y_t)) / N                        (0 at prompt positions)
    … back through the later layers to this layer's output:  δ = ∂L/∂y
    G = ∂L/∂Ŵ = Σ_tokens δ xᵀ
    STE:   ∂L/∂(W₀ + (α/r)·B·A) = G, masked to 0 where |·/s| > 6
    ∂L/∂B = (α/r) · G · Aᵀ,     ∂L/∂A = (α/r) · Bᵀ · G
update:   A, B ← AdamW step (FP32 masters);  W₀ unchanged
```

Example for one token: the target is "Paris" and the model gives it p = 0.2, so the token's loss is −log 0.2 = 1.61 and ∂L/∂z is −0.8 at "Paris" and +p elsewhere: push "Paris" up, everything else down. Without the clip mask, G·Aᵀ = δ (A x)ᵀ, so plain LoRA never forms the full d_out × d_in matrix G; QAT does form Ŵ, and the mask makes G itself necessary.

After training, the merged checkpoint Q(W₀ + (α/r)·B·A) **is** the model that was trained, so the merge loses nothing. One subtlety: at the start W₀ is on the grid and B·A is tiny, so Q(W₀ + BA) = W₀ and the forward pass doesn't change yet. The STE gradient still reaches A and B, and the adapter grows until it flips codes: training learns updates large enough to survive rounding.

### Dequantize to what

Generic QAT dequantizes back to BF16 or FP32 and runs a normal matmul. DeepSeek-V4 dequantizes into FP8 (E4M3 with 128×128 block scales) and runs its existing FP8 matmul, which works because power-of-two FP4 scales convert to FP8 exactly.

**Where the weights live.** In HBM, split across data-parallel ranks (ZeRO or a distributed optimizer), so each GPU holds a slice:

```text
HBM, persistent (each GPU's slice):   FP32 master w  +  Adam moments (BF16 in DeepSeek)
temporary, per layer, each step:      FP32 master → round to the FP4 grid → FP8 copy (in HBM) → GEMM, then freed
```

Unlike inference, nothing here is expanded tile by tile on chip: a kernel reads the FP32 master, quantizes it to the FP4 grid, expands it to FP8 and writes that copy to HBM as a temporary buffer, which the FP8 GEMM then reads (it is kept or recomputed for the backward pass, then freed). No FP4 codes are stored during training; FP4 is only the grid the values are snapped to.

### Why FP32 masters and not BF16

Not just a preference: BF16 masters would lose most updates. BF16 keeps 7 mantissa bits, so near 1.0 the gap between neighbours is 2⁻⁷ ≈ 0.0078. A typical update (learning rate × gradient) is around 10⁻⁴ to 10⁻⁵ of the weight:

```text
BF16:  1.0 + 0.0001  →  1.0       (less than half a step: lost; 1000 such updates: still 1.0)
FP32:  1.0 + 0.0001  =  1.0001    (step near 1 is 2⁻²³ ≈ 1.2e−7; 1000 updates: 1.1)
```

Any update smaller than about 0.4% of the weight disappears in BF16, so training stalls or drifts. DeepSeek saves memory where it is safe: in V3 (§3.3) the AdamW moments are BF16, because they only feed the update and are never accumulated into, while the master weights and gradient accumulators stay FP32. QAT makes this matter even more. The latent W has to **add up** many tiny updates until it crosses an FP4 rounding boundary, roughly 12% of the weight away. If each update is rounded off in BF16, W never gets there, the codes never flip, and nothing is learned.

If memory forces BF16:

- **BF16 + stochastic rounding.** This is where stochastic rounding shines: the same weight is rounded again and again over thousands of steps, so the errors cancel over time. (A one-shot merge rounds once, and the noise stays; see the LoRA section below.)
- **BF16 + Kahan summation.** A second BF16 buffer holds the bits lost at each step and adds them back later. That is 4 bytes per weight again, so it mainly lets everything stay in BF16 kernels rather than saving memory.

Most large runs keep FP32 masters.

## GPTQ: round, then fix up the rest

### The idea in plain words

**Splitting a bill in whole dollars.** Three friends owe $2.40, $2.40 and $2.20, $7.00 in total, and everyone must pay whole dollars.

```text
round each one on its own:     2 + 2 + 2 = $6        → the group is $1 short
one at a time, passing the leftover on:
  friend 1: 2.40 → pays 2, leftover +0.40 goes to friend 2
  friend 2: 2.40 + 0.40 = 2.80 → pays 3, leftover −0.20 goes to friend 3
  friend 3: 2.20 − 0.20 = 2.00 → pays 2
  total:    2 + 3 + 2 = $7  ✓
```

That is GPTQ's whole idea: **round the weights one at a time, and hand each rounding error to the weights not yet rounded, so the layer's output stays right.** The weights are the friends; the "total" is the layer output.

**The smart part.** In the bill, every friend counts equally toward the total. In a layer, some weights work together and some don't. Two weights whose inputs always move together can cover for each other: round one down, raise the other. Two weights whose inputs are unrelated can't help each other. GPTQ measures who works with whom from a small sample of real inputs, and splits each leftover in proportion to how much the remaining weights can make up for it.

A familiar picture is **dithering**: turning a grey photo into black and white dots. Rounding each pixel on its own gives ugly blotches; dithering rounds one pixel at a time and spreads its error to the neighbours, so from a distance the image still looks grey. GPTQ does the same with weights, except the "neighbours" are chosen by the data, not by position.

### The math

A layer computes y = W x. Each **row** w of W (one output) dots with the input x, whose entries x₁ … x_d are the input **features** (channels). Take the calibration inputs as a matrix X (d features × n tokens). GPTQ keeps the layer's output close to the original:

```text
min_Ŵ ‖(W − Ŵ) X‖²  =  Σ_rows (w − ŵ) H (w − ŵ)ᵀ,      H = X Xᵀ   (d × d)
```

H_jk = Σ_tokens x_j · x_k measures how much **input features** j and k move together across the calibration tokens. Note what is correlated: features (columns of W), not data samples, and what gets rounded: the **weights** w_j, never the inputs. Weight w_j multiplies feature x_j, so if x_j and x_k are correlated, an error in w_j can be cancelled by adjusting w_k. The rows don't interact, so each row is its own quadratic problem with the same H.

A **coordinate** is one entry of the row vector, one weight: "quantize coordinate q" means "round the q-th weight". (The word is from geometry: a row of d weights is a point in d-dimensional space.)

**One step** (from Optimal Brain Surgeon): round coordinate q, then choose the change Δ to the remaining weights that minimises the extra loss.

```text
δ_q = w_q − Q(w_q)                                     rounding error of weight q
min  Δ H Δᵀ   subject to   Δ_q = −δ_q
⇒  Δ = −δ_q · H⁻¹ e_q / [H⁻¹]_qq                       (Lagrange multiplier)
i.e.  w ← w − (δ_q / [H⁻¹]_qq) · H⁻¹[:, q]             loss increase = δ_q² / [H⁻¹]_qq
then drop q:   H⁻¹ ← H⁻¹ − H⁻¹[:, q] H⁻¹[q, :] / [H⁻¹]_qq
```

The error is shared among **all** remaining weights, in proportion to the entries of H⁻¹ (which measures how strongly two features are linked once the others are accounted for). Weights whose features are unrelated to feature q get almost nothing.

**Is H being updated?** Not as data. H = XXᵀ is computed once from the calibration set and never changes. The last line above is bookkeeping: once weight q is fixed it can no longer help, so it is removed from the problem, and H⁻¹ becomes the inverse over the weights still free. Only W is actually modified. Because GPTQ quantizes columns in the same left-to-right order for every row, that whole sequence of H⁻¹ updates is the same for all rows, and is precomputed once as a Cholesky factor. That is the `W[:, j + 1:] -= err[:, None] * Hinv[j, j + 1:]` line in the code above (with `Hinv` the Cholesky factor). Two more tricks: updates are applied lazily in blocks of 128 columns, and H gets a small damping term (1% of its mean diagonal).

**A two-weight example.** Integer grid, strongly correlated features:

```text
w = [0.4, 0.4],   H = [[1, 0.9], [0.9, 1]]   →   H⁻¹ = [[5.26, −4.74], [−4.74, 5.26]]

RTN:   ŵ = [0, 0],   loss = (w − ŵ) H (w − ŵ)ᵀ = 0.608

GPTQ:  round w₀: 0.4 → 0, δ = 0.4
       w₁ ← 0.4 − 0.4 · (−4.74 / 5.26) = 0.76  →  rounds to 1
       ŵ = [0, 1],   loss = 0.088                  ← 7× lower
```

Because the features are correlated, the output depends mostly on w₀ + w₁ = 0.8. Rounding each on its own makes the sum 0; GPTQ makes it 1. The first weight rounded down, so the second was pushed up to cover for it, exactly like the bill.

With these tricks GPTQ quantizes a 175B model in about 4 GPU hours, at 3–4 bits with little loss.
## AWQ: protect the channels that matter

AWQ (2023) starts from an observation: keeping about 1% of weight channels in FP16 recovers most of the accuracy, and the important channels are the ones whose **inputs** (activations) are large, not the ones with large weights. Mixed precision is awkward for kernels, so instead it scales:

```text
y = W x = (W · diag(s)) (diag(1/s) · x)
```

Multiplying a salient weight column by s > 1 makes it larger relative to the group scale, so it loses less to rounding; the activation is divided by s, which is folded into the previous layer, so the output is unchanged. The scales are s = (mean |x|)^α with α picked by a small grid search, which is the `awq` function above. No backprop, no weight updates.

| | GPTQ | AWQ |
|---|---|---|
| What it changes | The weights (error compensation) | Per-channel scales only |
| Uses | Second-order statistics (X Xᵀ) | Activation magnitudes |
| Speed to quantize | Slower | Faster |
| Calibration overfitting | More | Less |
| Typical format | INT4, group 128 | INT4, group 128 |

Both are W4A16: they speed up decoding, not prefill. vLLM, SGLang and TensorRT-LLM load both.

## GGUF: a file format, not a method

GGUF is the file format of llama.cpp, used for local and CPU inference. A file is a header (magic number, version, counts), typed key-value metadata (`general.architecture`, the tokenizer, hyperparameters), one info entry per tensor (name, shape, type, offset), then the tensor data aligned to 32 bytes so the whole file can be memory-mapped and loads almost instantly.

The quantization types you see on downloads:

| Name | What it is |
|---|---|
| Q8_0, Q4_0, Q5_0 | Legacy: blocks of 32 values, one scale each |
| Q4_1, Q5_1 | Legacy with a minimum as well as a scale (asymmetric) |
| Q2_K … Q6_K | K-quants: 256-value super-blocks whose sub-block scales are themselves quantized |
| Q4_K_M, Q4_K_S | A file-level mix: mostly Q4_K, with a few sensitive tensors at higher precision (more for _M than _S) |
| IQ1 … IQ4 | Lower-bit codebook formats, usually built with an importance matrix from calibration text |

## DeepSeek: training on the grid

**V3, FP8 training.** Every FP8 tensor uses E4M3, with scales per 1×128 activation tile and per 128×128 weight block. Fine-grained scales cover the dynamic range, so there is no need for E5M2's extra exponent bit. H800 tensor cores keep only about 14 bits when accumulating, so partial sums are promoted to FP32 every 128 elements.

**V4, FP4 QAT.** V4 was not trained in BF16 and quantized afterwards. Pre-training is FP8 mixed precision. Post-training adds MXFP4 QAT for the routed experts and the indexer: each step, the FP32 master weights are quantized to FP4, then expanded to FP8 for the matmul. That expansion is lossless, because power-of-two FP4 scales fit inside an FP8 block's exponent range, so the existing FP8 training stack is reused unchanged. RL rollouts use the real FP4 weights, so sampling matches deployment. The release is FP4 experts plus FP8 for almost everything else.

**NVIDIA's NVFP4 version** of V4-Flash is a re-encoding of that MXFP4 checkpoint, not a new training run. It is bit-exact because the per-tensor scale is a power of two, so every MXFP4 scale becomes an exact E4M3 value. It adds 4-bit activation scales for Blackwell, which DeepSeek's QAT never saw.

**V4.1, FP4 KV cache.** The KV cache is expanded before attention, so it doesn't need hardware FP4 support and can use the more accurate NVFP4-style layout (E2M1 with an E4M3 scale per 16 channels). It drops the tensor scale: the KV latent is RMS-normalized, so its values stay far below the 448 × 6 = 2,688 that block scales alone can reach. The indexer's FP4 matmuls stay on MXFP4, because more hardware runs them.

## A trap: merging a LoRA into a 4-bit checkpoint

A LoRA update D = (α/r)·B·A is small, about 1% of the weight. Merging it into an FP4 base means W' = Q(W + D). The base W is already exactly on the grid (it is dequantized FP4), and an FP4 grid step is about 25% of the weight, so a 1% change is almost always less than half a step. When we measured this on DeepSeek-V4-Flash's routed experts, the change that actually landed in the merged weights was mostly rounding noise (cosine with the trained update ≈ 0.12–0.16), while FP8 tensors kept roughly half of it.

### Round to nearest vs stochastic rounding

```text
RTN:  Q(x) = whichever of lo, hi is closer                         deterministic, biased
SR:   Q(x) = hi with probability (x − lo) / (hi − lo), else lo      a weighted coin flip
      E[Q(x)] = lo + (x − lo) = x                                   unbiased
```

A weight on grid point 2 (next point 3) gets Δ = 0.1, so the target is 2.1:

| | Result | On average | Squared error per weight |
|---|---|---|---|
| RTN | always 2 | 2.0: the update is **lost** | 0.1² = 0.01 |
| SR | 3 with probability 10%, else 2 | 2.1: the update is **kept** | 0.1·0.9² + 0.9·0.1² = **0.09** |

Over 1,000 such weights, the number that move under SR is binomial: mean n·p = 100, variance n·p·(1−p) = 90, standard deviation √90 ≈ 9.5. So a dot product with all inputs equal to 1 changes by about 100 ± 9.5 (RTN: 0). SR gets the total right by giving each weight 9× more squared error.

Why a "total" matters at all: a plain sum of weights means nothing, but each layer output is a weighted sum, y_i = Σ_j W_ij x_j (the example above is the case x_j = 1). Because the output is linear in W, E[R_ij] = D_ij gives E[Σ_j R_ij x_j] = Σ_j D_ij x_j for every input x: SR makes each layer's output change right **on average**. It says nothing about the noise, whose variance is Σ_j x_j² · Var(R_ij), and a single merged model lives with that noise.

**Unbiased is not the same as accurate.** SR is excellent when the same quantity is rounded again and again with fresh, independent errors, as in low-precision training (the NVFP4 paper uses it for gradients): the errors cancel over thousands of steps. A merge rounds once, and the noise is frozen into the weights.

### How to measure a merge

With W the FP4 base, D the trained update and R = Q(W + D) − W what the merge actually changed (ideally R = D):

| Metric | Formula | Meaning | Ideal |
|---|---|---|---|
| Share of D kept | ⟨R, D⟩ / ⟨D, D⟩ | How much of R lies along D, in units of D | 1 |
| cos(R, D) | ⟨R, D⟩ / (‖R‖ ‖D‖) | Does the applied change point the same way as the trained one? Ignores size | 1 |
| ‖R − D‖ / ‖D‖ | relative error | How far the applied change is from the trained one | **0** |
| Output-change error | ‖X(R − D)ᵀ‖ / ‖X Dᵀ‖ | The same, measured on layer outputs for real inputs X | **0** |

A cosine near 1 does not mean the change dominates the merged weights: those are W + R, still about 99% W. It means the edit you applied has the direction of the edit you trained, so the merged model behaves like the trained adapter. The metrics have to be read together: ‖R − D‖/‖D‖ = 1 for both R = 0 (did nothing; cos = 0, share = 0) and R = 2D (right direction, twice the size; cos = 1, share = 2). A value of 1 is the "do nothing" baseline; anything above 1 is worse than not merging.

"1%" and the "weights changed" column measure different things: 1% is the **size** of the update (‖D‖ = 0.01·‖W‖, Frobenius norm; D is dense, so every weight changes a little), while "weights changed" **counts** weights whose stored FP4 code moved. Under SR a weight jumps with probability |D_ij| / gap, so with |D_ij| around 2.5% of a gap, about 2.5% of weights jump, each by a full gap.

### What GPTQ requantization means

**Quantization** turns BF16 weights into FP4 codes and scales. **Requantization** quantizes again weights that were already quantized and then changed, and a LoRA merge is exactly that:

```text
W     = dequantize(FP4 base)          already on the grid
T     = W + D                         target after adding the LoRA, off the grid
merge = Q(T)                          must go back to FP4 to ship
```

The only question is how Q(T) rounds. **RTN** rounds each weight of T on its own (what a typical merge script does). **GPTQ requantization** runs the GPTQ algorithm on T instead: column by column, pushing each rounding error onto the columns not yet rounded, using that layer's real input activations X:

```text
min over FP4 codes Q:   ‖(Q − (W + D)) X‖²        keep the layer's output close to the merged target
```

In practice:

1. Run a few hundred calibration samples through the model and record the input X to every linear layer.
2. For each layer, build H = XXᵀ and run GPTQ on W + D instead of rounding to nearest.
3. Write out the new FP4 codes, reusing the existing block scales or recomputing them.

Two snags for an MoE model such as DeepSeek-V4-Flash. Each expert needs its own H, built only from the tokens routed to it, so rarely used experts may see too few calibration tokens for a reliable H unless the calibration set is large. And most GPTQ tooling was built for INT4 with groups of 128; support for writing MXFP4 or NVFP4 targets is patchier.

### A simulation: RTN vs SR vs GPTQ

A 1024×1024 MXFP4 weight (block scales kept fixed), a rank-16 update, correlated input features. GPTQ requantizes W + D using 4,096 calibration inputs; all output errors are measured on 4,096 different inputs.

```python
"""Merge a small LoRA update into an MXFP4 weight: round-to-nearest vs stochastic rounding vs GPTQ."""
import torch
torch.manual_seed(0)
G = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.]); G = torch.cat([-G.flip(0)[:-1], G])   # E2M1 grid, sorted

def neighbours(x):                       # grid points just below and above x
    i = torch.searchsorted(G, x.clamp(-6, 6).contiguous()).clamp(1, len(G) - 1)
    return G[i - 1], G[i]
def rtn(x):
    lo, hi = neighbours(x); return torch.where(x - lo < hi - x, lo, hi)
def sr(x):                               # round up with probability = distance from lo / gap
    lo, hi = neighbours(x); p = (x - lo) / (hi - lo)
    return torch.where(torch.rand_like(x) < p, hi, lo)

out_f, in_f, r = 1024, 1024, 16
W0 = torch.randn(out_f, in_f) * 0.02
S = (2.0 ** torch.ceil(torch.log2(W0.reshape(out_f, -1, 32).abs().amax(-1, keepdim=True) / 6))
     ).expand(-1, -1, 32).reshape(out_f, in_f)                    # MXFP4 block scales, kept fixed
q = lambda W, f: f(W / S) * S
W = q(W0, rtn)                                                    # the FP4 base: already on the grid
B, A = torch.randn(out_f, r), torch.randn(r, in_f)
import sys
SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01            # |D| / |W|
D = B @ A; D *= SIZE * W.norm() / D.norm()                        # LoRA update, 1% of |W| by default
mix = torch.randn(in_f, in_f) / in_f ** 0.5
X_cal, X_test = torch.randn(4096, in_f) @ mix, torch.randn(4096, in_f) @ mix   # correlated input features

def gptq(T, X):                          # round T column by column, pushing each error onto later columns
    T = T.clone(); Q = torch.zeros_like(T)
    H = X.T @ X; H += 0.01 * H.diag().mean() * torch.eye(len(H))
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    for j in range(T.shape[1]):
        Q[:, j] = rtn(T[:, j] / S[:, j]) * S[:, j]
        err = (T[:, j] - Q[:, j]) / U[j, j]
        T[:, j + 1:] -= err[:, None] * U[j, j + 1:]
    return Q

for name, merged in [("RTN", q(W + D, rtn)), ("stochastic", q(W + D, sr)), ("GPTQ", gptq(W + D, X_cal))]:
    R = merged - W                                                # what the merge actually changed
    kept = (R * D).sum() / (D * D).sum()
    cos = torch.nn.functional.cosine_similarity(R.flatten(), D.flatten(), 0)
    out_err = (X_test @ (R - D).T).norm() / (X_test @ D.T).norm() # on inputs GPTQ never saw
    print(f"{name:10s} weights changed {(R != 0).float().mean():5.1%}   share of D kept {kept:6.1%}   "
          f"cos(R,D) {cos:.2f}   |R-D|/|D| {(R - D).norm() / D.norm():5.2f}   output-change error {out_err:5.2f}")
```

```text
|D| = 1% of |W|
RTN        weights changed  0.0%   share of D kept   0.0%   cos(R,D) 0.00   |R-D|/|D|  1.00   output-change error  1.00
stochastic weights changed  2.5%   share of D kept  99.3%   cos(R,D) 0.19   |R-D|/|D|  5.27   output-change error  5.29
GPTQ       weights changed  0.0%   share of D kept   0.0%   cos(R,D) 0.00   |R-D|/|D|  1.00   output-change error  1.00

|D| = 5% of |W|
RTN        weights changed  1.7%   share of D kept  16.9%   cos(R,D) 0.32   |R-D|/|D|  0.97   output-change error  0.97
stochastic weights changed 12.6%   share of D kept  98.8%   cos(R,D) 0.41   |R-D|/|D|  2.18   output-change error  2.19
GPTQ       weights changed  6.1%   share of D kept  36.7%   cos(R,D) 0.33   |R-D|/|D|  1.23   output-change error  0.98

|D| = 20% of |W|
RTN        weights changed 37.9%   share of D kept  92.1%   cos(R,D) 0.86   |R-D|/|D|  0.54   output-change error  0.55
stochastic weights changed 44.2%   share of D kept  99.0%   cos(R,D) 0.78   |R-D|/|D|  0.80   output-change error  0.81
GPTQ       weights changed 45.3%   share of D kept  94.9%   cos(R,D) 0.75   |R-D|/|D|  0.84   output-change error  0.43
```

What it shows:

- **At 1% (a realistic LoRA), no rounding method helps.** RTN changes nothing. SR keeps 99% of D on average but adds noise 5× the size of the update, worse than not merging. GPTQ also changes nothing, and that is the right answer: with the update far below the grid's resolution, leaving the base alone gives the smallest output error of anything on the grid.
- **At 5%**, RTN and GPTQ are only as good as doing nothing; SR is still worse than that.
- **At 20%**, the update is big enough to represent, and GPTQ wins clearly on outputs (0.43 vs 0.55 for RTN), even though its weight-space error is the largest: it optimises what the layer computes, not how close each weight is.

The real DSv4 merge kept 30–80% of D with RTN and changed 3–5% of weights, more than the 1% row here, because a real LoRA update has a few large entries and the merge script recomputes block scales. A uniform random low-rank update has neither.

### So how should you merge into FP4?

1. **Don't requantize.** Merge into the BF16 checkpoint, or serve the base plus a runtime LoRA.
2. **QAT LoRA.** Train the adapter through Q(W₀ + (α/r)·B·A) (see QAT above); the merged FP4 model is then exactly the trained one, and the adapter learns updates large enough to survive rounding.
3. **GPTQ-style requantization** helps once the update is a sizeable fraction of a grid step, not for a typical 1% LoRA.
4. **Stochastic rounding** is unbiased but noisy; for a one-shot merge, check the merged model on held-out data before trusting it.
## Cheat sheet

| Term | One line |
|---|---|
| W4A16 | 4-bit weights, BF16 math: faster decode, same prefill |
| W8A8 / FP8 | Low-precision matmul: faster prefill and training |
| Group / block size | Values sharing one scale; smaller = better with outliers, more overhead |
| E2M1 (FP4) | ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} |
| E4M3 (FP8) | ±2^(E−7)(1 + M/8), max 448 |
| E8M0 | 2^(E−127): a pure-exponent scale |
| MXFP4 | FP4 + E8M0 per 32 → 4.25 bits |
| NVFP4 | FP4 + E4M3 per 16 + FP32 per tensor → 4.5 bits |
| PTQ / QAT | Quantize after training / train on the grid |
| GPTQ | Column by column, push rounding error onto the rest via H⁻¹ |
| STE | Backward pass treats round() as identity (zero gradient where clipped) |
| Stochastic rounding | Round up with probability = distance; unbiased but noisy, good for repeated rounding |
| AWQ | Scale up channels with large activations before rounding |
| GGUF | llama.cpp file format; Q4_K_M = mostly 4-bit k-quants |

## References

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Rouhani et al., *Microscaling Data Formats for Deep Learning*, [arXiv:2310.10537](https://arxiv.org/abs/2310.10537)
- NVIDIA, *Pretraining Large Language Models with NVFP4*, [arXiv:2509.25149](https://arxiv.org/abs/2509.25149)
- GGUF specification, [ggml docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-AI, *DeepSeek-V4*, [arXiv:2606.19348](https://arxiv.org/abs/2606.19348); *DeepSeek-V4.1-Flash*, [arXiv:2609.19969](https://arxiv.org/abs/2609.19969)
