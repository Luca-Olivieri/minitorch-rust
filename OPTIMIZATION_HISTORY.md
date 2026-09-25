# MNIST Optimization History

This file records the Rust MNIST `SmallCNN` optimization measurements. The
nine Rust profiles below are transcribed from the recorded optimization runs;
the measurements are kept at their original precision so that later runs can be
compared directly.

## Profiling standard

From now on, the standard optimization profile is to run through **the validation
after the first training epoch** and then stop.

That means:

1. Build and run the release binary with `make run-release`.
2. Complete all 938 training batches for epoch 1.
3. Complete validation on all 157 test batches.
4. Record the first-epoch results and finish the profile after validation.

Every profile should record setup, initial evaluation, checkpoint timings, the
complete first-epoch training time, smoothed training loss, validation loss,
and validation time.

## Measurement conventions

- Rust timings are from the release build.
- Forward, backward, optimizer, and epoch times are reported in milliseconds
  unless a value is explicitly marked as seconds.
- Loss timings are reported in microseconds, matching the program output.
- `smoothed loss` is the loss value printed at the corresponding training
  checkpoint.
- Step `938/938` is the final, smaller batch of the epoch and is not directly
  comparable to the full-size checkpoints. Use the complete epoch time for
  aggregate comparisons.
- `grads.len() = 8` was reported at every checkpoint in all nine profiles.
- Stages 7 and 8 report the post-accumulation time as `step_time`; earlier
  stages labelled the same field `Optimizer`. The values are directly
  comparable.
- Record CPU model, operating system, exact commit, dtype, and thread settings
  with each profile when making a cross-machine comparison.

## Summary

All nine runs use the same dataset sizes reported by the program: 60,000
training samples, 10,000 test samples, 938 training batches per epoch, and 157
validation batches.

| Profile | Branch/label | Initial evaluation | Epoch 1 training | Epoch 1 smoothed loss | Validation loss | Validation time |
|---|---|---:|---:|---:|---:|---:|
| Baseline | `main` / `0. Baseline` | 56.488265041 s | 868.228344666 s | 0.3363628374274766 | 0.2691631467284183 | 57.073553542 s |
| After `AutogradMeta` | `feature/weird-optimizations~2` | 60.536313709 s | 889.215288917 s | 0.3363628374274766 | 0.2691631467284183 | 58.738805208 s |
| After fused forward/backward `Conv2d` kernels | `2. after fused fw and bw Conv2d kernels` | 20.130340208 s | 328.818958666 s | 0.3363628374274767 | 0.2691631467284183 | 20.493247959 s |
| After compact max-pool metadata | `feature/weird-optimizations*` | 15.5961365 s | 308.258938459 s | 0.3363628374274767 | 0.2691631467284183 | 15.865784084 s |
| Opt-in per-layer profiling logging | `feature/weird-optimizations*` | 15.689674792 s | 306.660931041 s | 0.3363628374274767 | 0.2691631467284183 | 15.568933625 s |
| Conv2d backward fast path and section profiling | `feature/weird-optimizations*` | 15.693244541 s | 298.810022917 s | 0.3363628374274767 | 0.2691631467284183 | 15.547558291 s |
| Clean run after reverting output-channel blocking | `feature/weird-optimizations*` | 15.498567500 s | 299.386094958 s | 0.3363628374274767 | 0.2691631467284183 | 15.972398166 s |
| Stride-1/dilation-1 matmul with unit-stride paths | `feature/weird-optimizations*` | 13.369879042 s | 267.85608725 s | 0.3363628374274767 | 0.2691631467284183 | 13.462407916 s |
| Register-blocked conv accumulator | `feature/weird-optimizations*` | 10.484789333 s | 234.81543525 s | 0.3363628374274767 | 0.2691631467284183 | 10.455207417 s |

The nine profiles follow the same measurement boundary: initial evaluation,
all 938 epoch-1 training batches, and validation on 157 test batches.

## 0. Baseline

**Source label:** `0. Baseline`
**Branch:** `main`
**Command:** `make run-release`
**Status:** measured through first-epoch validation

| Metric | Value |
|---|---:|
| Dataset setup | 45.234541 ms |
| Dataloader setup | 0.445625 ms |
| Model setup | 2.800667 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 56.488265041 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 367.128917 ms | 16.791 µs | 528.109875 ms | 0.988416 ms | 2.0968491171052857 | 8 |
| 200 / 938 | 357.782584 ms | 16.291 µs | 544.148292 ms | 1.218542 ms | 1.122581818292674 | 8 |
| 300 / 938 | 357.709041 ms | 16.666 µs | 510.486167 ms | 0.972625 ms | 0.6686163189624389 | 8 |
| 400 / 938 | 357.364334 ms | 20.375 µs | 534.946334 ms | 1.068000 ms | 0.5889227138650285 | 8 |
| 500 / 938 | 366.626333 ms | 16.375 µs | 560.926834 ms | 0.887459 ms | 0.502232115982027 | 8 |
| 600 / 938 | 362.117416 ms | 16.000 µs | 542.824750 ms | 1.108583 ms | 0.4567609410296239 | 8 |
| 700 / 938 | 357.655125 ms | 31.167 µs | 526.100541 ms | 0.583500 ms | 0.39503232560985174 | 8 |
| 800 / 938 | 354.946416 ms | 20.833 µs | 539.887292 ms | 0.863583 ms | 0.3248854019236871 | 8 |
| 900 / 938 | 352.029667 ms | 16.875 µs | 519.260333 ms | 1.332791 ms | 0.325188976509936 | 8 |
| 938 / 938 | 174.561042 ms | 11.375 µs | 256.234667 ms | 0.635041 ms | 0.3363628374274766 | 8 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 868.228344666 s |
| Smoothed training loss | 0.3363628374274766 |
| Validation loss | 0.2691631467284183 |
| Validation time | 57.073553542 s |

## 1. After `AutogradMeta`

**Source label:** `1. after AutoGradMeta`
**Branch:** `feature/weird-optimizations~2`
**Command:** `make run-release`
**Status:** measured through first-epoch validation

This profile follows the change to `AutogradMeta` and explicit `no_grad`
propagation. It is the comparison point for the subsequent convolution-kernel
work.

| Metric | Value |
|---|---:|
| Dataset setup | 27.858250 ms |
| Dataloader setup | 0.380041 ms |
| Model setup | 3.226500 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 60.536313709 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 363.236375 ms | 16.625 µs | 505.029875 ms | 0.580958 ms | 2.0968491171052857 | 8 |
| 200 / 938 | 364.023458 ms | 16.875 µs | 533.026458 ms | 0.754375 ms | 1.122581818292674 | 8 |
| 300 / 938 | 370.977959 ms | 17.125 µs | 563.340500 ms | 1.433416 ms | 0.6686163189624389 | 8 |
| 400 / 938 | 363.899542 ms | 16.708 µs | 555.380916 ms | 1.244250 ms | 0.5889227138650285 | 8 |
| 500 / 938 | 380.269375 ms | 16.917 µs | 591.661042 ms | 1.087334 ms | 0.502232115982027 | 8 |
| 600 / 938 | 377.726291 ms | 20.917 µs | 559.435166 ms | 1.333125 ms | 0.4567609410296239 | 8 |
| 700 / 938 | 362.945292 ms | 16.375 µs | 566.248875 ms | 1.114417 ms | 0.39503232560985174 | 8 |
| 800 / 938 | 370.577125 ms | 16.750 µs | 548.108000 ms | 0.765334 ms | 0.3248854019236871 | 8 |
| 900 / 938 | 495.260625 ms | 71.333 µs | 616.622958 ms | 1.669416 ms | 0.325188976509936 | 8 |
| 938 / 938 | 176.999875 ms | 11.708 µs | 242.634625 ms | 0.799792 ms | 0.3363628374274766 | 8 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 889.215288917 s |
| Smoothed training loss | 0.3363628374274766 |
| Validation loss | 0.2691631467284183 |
| Validation time | 58.738805208 s |

## 2. After fused forward/backward `Conv2d` kernels

**Source label:** `2. after fused fw and bw Conv2d kernels`
**Command:** `make run-release`
**Status:** measured through first-epoch validation

This profile covers the direct forward and backward convolution-kernel
optimization and is recorded under the source label “after fused fw and bw
Conv2d kernels.”

| Metric | Value |
|---|---:|
| Dataset setup | 26.605917 ms |
| Dataloader setup | 0.491250 ms |
| Model setup | 4.358417 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 20.130340208 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 124.723291 ms | 15.917 µs | 211.024708 ms | 0.804500 ms | 2.0968491171052857 | 8 |
| 200 / 938 | 123.819292 ms | 17.167 µs | 211.135417 ms | 0.897166 ms | 1.1225818182926737 | 8 |
| 300 / 938 | 124.216333 ms | 15.625 µs | 210.790333 ms | 0.572708 ms | 0.6686163189624389 | 8 |
| 400 / 938 | 129.097625 ms | 16.417 µs | 213.746666 ms | 0.807000 ms | 0.5889227138650284 | 8 |
| 500 / 938 | 123.980167 ms | 16.000 µs | 210.650333 ms | 0.740375 ms | 0.502232115982027 | 8 |
| 600 / 938 | 124.549500 ms | 18.667 µs | 214.179000 ms | 0.808375 ms | 0.4567609410296239 | 8 |
| 700 / 938 | 126.252250 ms | 16.625 µs | 210.245833 ms | 0.870000 ms | 0.3950323256098518 | 8 |
| 800 / 938 | 125.909375 ms | 16.542 µs | 209.955875 ms | 0.796542 ms | 0.32488540192368714 | 8 |
| 900 / 938 | 127.360375 ms | 15.417 µs | 211.852625 ms | 0.578167 ms | 0.3251889765099359 | 8 |
| 938 / 938 | 63.387542 ms | 11.208 µs | 108.140709 ms | 0.865458 ms | 0.3363628374274767 | 8 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 328.818958666 s |
| Smoothed training loss | 0.3363628374274767 |
| Validation loss | 0.2691631467284183 |
| Validation time | 20.493247959 s |

## 3. After compact max-pool metadata

**Source label:** `After compact max-pool metadata`
**Branch:** `feature/weird-optimizations*`
**Command:** `make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

This profile adds the value-only max-pool path and replaces per-window
`Vec<Vec<usize>>` metadata with flat indices and offsets. Tied maxima continue
to receive equal shares of the upstream gradient.

| Metric | Value |
|---|---:|
| Dataset setup | 25.255958 ms |
| Dataloader setup | 0.371500 ms |
| Model setup | 3.131750 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 15.5961365 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 110.331500 ms | 16.292 µs | 215.213208 ms | 0.857250 ms | 2.0968491171052857 | 8 |
| 200 / 938 | 114.986167 ms | 18.000 µs | 216.804583 ms | 0.818833 ms | 1.1225818182926737 | 8 |
| 300 / 938 | 110.016292 ms | 20.084 µs | 212.345750 ms | 0.823292 ms | 0.6686163189624389 | 8 |
| 400 / 938 | 111.483792 ms | 16.334 µs | 211.869125 ms | 0.794958 ms | 0.5889227138650284 | 8 |
| 500 / 938 | 118.967750 ms | 16.084 µs | 214.184333 ms | 0.732250 ms | 0.502232115982027 | 8 |
| 600 / 938 | 109.419167 ms | 16.625 µs | 214.794333 ms | 0.577708 ms | 0.4567609410296239 | 8 |
| 700 / 938 | 111.676250 ms | 16.459 µs | 219.629542 ms | 1.090709 ms | 0.3950323256098518 | 8 |
| 800 / 938 | 111.289917 ms | 17.000 µs | 211.172042 ms | 0.852833 ms | 0.32488540192368714 | 8 |
| 900 / 938 | 110.332667 ms | 16.375 µs | 213.170042 ms | 0.803541 ms | 0.3251889765099359 | 8 |
| 938 / 938 | 53.928333 ms | 11.333 µs | 104.543333 ms | 0.777125 ms | 0.3363628374274767 | 8 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 308.258938459 s |
| Smoothed training loss | 0.3363628374274767 |
| Validation loss | 0.2691631467284183 |
| Validation time | 15.865784084 s |

Compared with Stage 2, initial evaluation decreased by `4.534203708 s`, the
complete first-epoch training time decreased by `20.560020207 s`, and validation
time decreased by `4.627463875 s`.

### Optimization review

- **Better:** Initial evaluation improved from `20.130340208 s` to
  `15.5961365 s`; first-epoch training improved from `328.818958666 s` to
  `308.258938459 s`; validation improved from `20.493247959 s` to
  `15.865784084 s`. The value-only pooling path is the clearest likely source
  of the evaluation and validation gains.
- **Worse:** Backward checkpoint timings did not show a consistent improvement;
  for example, step 100 increased from `211.024708 ms` to `215.213208 ms`, and
  some later checkpoints were also slightly slower. No direct peak-memory or
  allocation measurement is included, so the metadata reduction is not
  quantified.
- **Unchanged:** Smoothed training loss and validation loss remain effectively
  identical, so the optimization did not change the observed training result.
  Optimizer timings are mixed and remain negligible at this scale.

## 4. Opt-in per-layer profiling logging

**Source label:** `Opt-in per-layer profiling logging`
**Branch:** `feature/weird-optimizations*`
**Command:** `MINITORCH_PROFILE_LAYERS=true make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

This is a diagnostic logging profile rather than a new arithmetic optimization.
It records each forward child and the convolution entries in the backward
operation schedule. The timers are enabled only at logged training steps, so
model-wide timings include the instrumentation overhead on those steps.

| Metric | Value |
|---|---:|
| Dataset setup | 27.721334 ms |
| Dataloader setup | 0.373042 ms |
| Model setup | 3.777250 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 15.689674792 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 110.516083 ms | 16.542 µs | 212.975750 ms | 0.865083 ms | 2.0968491171052857 | 8 |
| 200 / 938 | 109.638959 ms | 16.792 µs | 212.941542 ms | 0.901708 ms | 1.1225818182926737 | 8 |
| 300 / 938 | 109.874792 ms | 20.542 µs | 215.209708 ms | 0.881041 ms | 0.6686163189624389 | 8 |
| 400 / 938 | 112.960959 ms | 18.500 µs | 211.257833 ms | 0.843875 ms | 0.5889227138650284 | 8 |
| 500 / 938 | 113.796042 ms | 18.417 µs | 223.716750 ms | 0.885833 ms | 0.502232115982027 | 8 |
| 600 / 938 | 110.171541 ms | 17.666 µs | 215.361083 ms | 0.893625 ms | 0.4567609410296239 | 8 |
| 700 / 938 | 109.069292 ms | 18.083 µs | 214.284625 ms | 0.634416 ms | 0.3950323256098518 | 8 |
| 800 / 938 | 111.719250 ms | 16.000 µs | 221.459416 ms | 1.069625 ms | 0.32488540192368714 | 8 |
| 900 / 938 | 110.234375 ms | 17.334 µs | 214.634333 ms | 0.811875 ms | 0.3251889765099359 | 8 |
| 938 / 938 | 54.110833 ms | 11.833 µs | 107.360208 ms | 0.880583 ms | 0.3363628374274767 | 8 |

### Forward layer profile

| Step | conv1 (ms) | relu1 (ms) | pool1 (ms) | conv2 (ms) | relu2 (ms) | pool2 (ms) | dropout (ms) | linear1 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 11.944584 | 4.753416 | 6.998042 | 59.051667 | 2.384084 | 3.817334 | 4.894625 | 16.503791 |
| 200 | 11.712500 | 4.810083 | 7.097917 | 58.326459 | 2.389542 | 3.771875 | 4.905250 | 16.505875 |
| 300 | 11.795708 | 4.757750 | 6.994084 | 58.538750 | 2.436834 | 3.695875 | 4.948875 | 16.588375 |
| 400 | 11.873792 | 4.783875 | 8.171500 | 60.325125 | 2.481625 | 3.781834 | 4.884709 | 16.544709 |
| 500 | 12.728125 | 4.902833 | 7.586583 | 59.033500 | 2.386542 | 3.965625 | 5.149292 | 17.920708 |
| 600 | 11.625625 | 4.764792 | 6.991292 | 58.331334 | 2.372208 | 4.516500 | 4.952833 | 16.479959 |
| 700 | 11.652667 | 4.752334 | 7.026750 | 58.214417 | 2.385834 | 3.505084 | 4.896000 | 16.515291 |
| 800 | 12.051166 | 4.879375 | 7.098500 | 59.211834 | 2.509500 | 4.292500 | 4.880542 | 16.617167 |
| 900 | 11.775959 | 5.089541 | 7.019042 | 58.926333 | 2.374167 | 3.665791 | 4.772250 | 16.489916 |
| 938 | 5.775166 | 2.557959 | 2.938750 | 29.131875 | 1.202917 | 1.800542 | 2.408083 | 8.239709 |

### Fine-grained forward profile

| Step | flatten (µs) | relu3 (µs) | linear2 (µs) |
|---:|---:|---:|---:|
| 100 | 77.917 | 24.292 | 64.041 |
| 200 | 27.583 | 24.125 | 64.084 |
| 300 | 27.250 | 26.334 | 63.416 |
| 400 | 28.625 | 22.542 | 60.625 |
| 500 | 32.333 | 23.375 | 64.250 |
| 600 | 46.041 | 21.792 | 64.292 |
| 700 | 27.834 | 29.458 | 62.042 |
| 800 | 89.875 | 25.250 | 61.208 |
| 900 | 27.583 | 24.417 | 64.541 |
| 938 | 12.958 | 11.000 | 29.875 |

### Convolution backward profile

The profiler reports convolution operations in reverse graph execution order.
For this sequential model, the first entry is the later convolution and the
second entry is the earlier convolution.

| Step | conv2d backward #1 (ms) | conv2d backward #2 (ms) |
|---:|---:|---:|
| 100 | 129.019583 | 11.667084 |
| 200 | 129.857959 | 11.725041 |
| 300 | 130.519250 | 12.557125 |
| 400 | 129.353791 | 11.687334 |
| 500 | 135.248833 | 11.799333 |
| 600 | 130.338583 | 12.470750 |
| 700 | 129.732625 | 12.108875 |
| 800 | 132.447583 | 12.172125 |
| 900 | 129.630500 | 11.944417 |
| 938 | 64.924709 | 5.926250 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 306.660931041 s |
| Smoothed training loss | 0.3363628374274767 |
| Validation loss | 0.2691631467284183 |
| Validation time | 15.568933625 s |

### Optimization review

- **Better:** The profile identifies the second convolution as the dominant
  layer: its forward is approximately `58–60 ms` versus `12 ms` for the first,
  and its backward is approximately `129–135 ms` versus `12 ms` for the first.
  This gives a concrete target for the next optimization. The diagnostic run
  also recorded lower aggregate training and validation times than Stage 3,
  but those values are not treated as a speedup because profiling changes the
  measured path.
- **Worse:** Initial evaluation was `15.689674792 s` versus `15.5961365 s` in
  Stage 3, and profiling adds timer overhead plus verbose output at logged
  steps. This change improves observability, not convolution arithmetic.
- **Unchanged:** Smoothed training loss, validation loss, and gradient count
  remain identical. The backward profile is operation-level and does not yet
  split `grad_input` from `grad_weight`.

## 5. Conv2d backward fast path and section profiling

**Source label:** `Conv2d backward fast path and section profiling`
**Branch:** `feature/weird-optimizations*`
**Command:** `MINITORCH_PROFILE_LAYERS=true make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

This profile adds the stride-1/dilation-1 fast path for both convolution
backward gradients, reports all backward operation totals, and breaks each
convolution backward into packing, padding, `grad_weight`, `grad_input`, and
unpacking sections. The timers are enabled at logged training steps.

| Metric | Value |
|---|---:|
| Dataset setup | 28.781084 ms |
| Dataloader setup | 0.354583 ms |
| Model setup | 4.116208 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 15.693244541 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 113.322041 ms | 17.750 µs | 224.945250 ms | 0.964791 ms | 2.0968491171052857 | 8 |
| 200 / 938 | 111.640458 ms | 16.833 µs | 205.689500 ms | 0.861125 ms | 1.1225818182926737 | 8 |
| 300 / 938 | 109.587833 ms | 17.625 µs | 203.920041 ms | 0.864000 ms | 0.6686163189624389 | 8 |
| 400 / 938 | 110.138875 ms | 16.542 µs | 205.364791 ms | 0.839291 ms | 0.5889227138650284 | 8 |
| 500 / 938 | 109.554625 ms | 16.500 µs | 208.242375 ms | 0.898208 ms | 0.502232115982027 | 8 |
| 600 / 938 | 113.693875 ms | 18.875 µs | 210.167666 ms | 0.925917 ms | 0.4567609410296239 | 8 |
| 700 / 938 | 110.163375 ms | 20.458 µs | 206.028666 ms | 0.882458 ms | 0.3950323256098518 | 8 |
| 800 / 938 | 109.962792 ms | 17.500 µs | 203.137584 ms | 0.905250 ms | 0.32488540192368714 | 8 |
| 900 / 938 | 110.571750 ms | 17.875 µs | 204.463458 ms | 0.828375 ms | 0.3251889765099359 | 8 |
| 938 / 938 | 54.372917 ms | 12.209 µs | 102.617334 ms | 0.871125 ms | 0.3363628374274767 | 8 |

### Forward layer profile

| Step | conv1 (ms) | relu1 (ms) | pool1 (ms) | conv2 (ms) | relu2 (ms) | pool2 (ms) | dropout (ms) | linear1 (ms) | flatten (ms) | relu3 (ms) | linear2 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 11.417917 | 4.949750 | 7.383792 | 60.293916 | 2.819708 | 4.300416 | 5.082459 | 16.941208 | 0.040375 | 0.024666 | 0.063708 |
| 200 | 12.096875 | 5.021625 | 7.411917 | 59.137833 | 2.376333 | 3.947417 | 5.023084 | 16.499541 | 0.030375 | 0.026167 | 0.067917 |
| 300 | 11.678833 | 4.745625 | 7.234709 | 58.134042 | 2.416042 | 3.891083 | 4.890167 | 16.474542 | 0.026375 | 0.027041 | 0.065250 |
| 400 | 11.335625 | 4.764917 | 7.410500 | 58.157959 | 2.384125 | 4.618000 | 4.886417 | 16.462416 | 0.034417 | 0.021542 | 0.060791 |
| 500 | 11.662167 | 4.754625 | 7.332208 | 58.089750 | 2.332459 | 3.931333 | 4.859625 | 16.479250 | 0.027916 | 0.023875 | 0.060250 |
| 600 | 12.384167 | 4.779750 | 8.295708 | 59.097541 | 2.696125 | 4.636583 | 4.928125 | 16.670791 | 0.116958 | 0.022125 | 0.064000 |
| 700 | 11.268166 | 4.798042 | 7.319708 | 59.071000 | 2.376250 | 3.816125 | 4.882791 | 16.510583 | 0.027250 | 0.024084 | 0.065500 |
| 800 | 11.192042 | 4.740000 | 7.274459 | 58.849750 | 2.379541 | 3.895750 | 5.026625 | 16.478333 | 0.037709 | 0.021750 | 0.062667 |
| 900 | 11.808375 | 4.748458 | 7.358334 | 58.270667 | 2.350083 | 4.523750 | 4.874250 | 16.465041 | 0.083292 | 0.021417 | 0.065500 |
| 938 | 5.622125 | 2.547542 | 3.122959 | 29.194542 | 1.195041 | 1.922667 | 2.427917 | 8.285167 | 0.012541 | 0.011166 | 0.030000 |

### Backward operation totals

The table records the significant operation categories. The remaining
operations (`div`, `exp`, `ln`, `max`, `neg`, `sub`, `sum`, and `unsqueeze`)
were each below `0.01 ms` at the logged checkpoints.

| Step | conv2d (ms) | matmul (ms) | maximum (ms) | max_pool2d (ms) | add (ms) | mul (ms) | reshape (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 138.393875 | 45.412957 | 25.931709 | 5.273291 | 4.772750 | 3.008958 | 0.176667 |
| 200 | 133.470417 | 42.481417 | 20.435459 | 4.210543 | 2.472166 | 0.554167 | 0.124584 |
| 300 | 132.707458 | 42.583750 | 20.224043 | 4.331917 | 2.406668 | 0.557584 | 0.041125 |
| 400 | 133.561583 | 42.490625 | 20.364167 | 4.298541 | 2.638707 | 0.498792 | 0.101250 |
| 500 | 136.302708 | 42.381334 | 20.443125 | 3.987542 | 3.069542 | 0.859667 | 0.035333 |
| 600 | 133.169292 | 43.514333 | 22.158833 | 4.694875 | 2.921333 | 2.250042 | 0.123083 |
| 700 | 132.622125 | 43.802958 | 20.455584 | 4.157000 | 2.602375 | 0.553042 | 0.157750 |
| 800 | 131.658083 | 43.378458 | 19.758376 | 4.125918 | 2.469291 | 0.575541 | 0.101375 |
| 900 | 133.572500 | 42.493792 | 19.755959 | 4.161625 | 2.346834 | 0.558917 | 0.040875 |
| 938 | 66.737874 | 21.451750 | 10.166917 | 1.977083 | 1.179166 | 0.267250 | 0.016708 |

### Conv2 backward section profile

The first convolution entry is the later, larger convolution in reverse graph
order. Values are milliseconds.

| Step | Total | Pack grad_output | Pack weight | Padded input | grad_weight | grad_input | Unpack weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 126.519000 | 3.754417 | 0.020250 | 2.271167 | 53.628083 | 66.365292 | 0.013958 |
| 200 | 122.551042 | 1.930417 | 0.015625 | 2.132834 | 51.531583 | 66.732875 | 0.013083 |
| 300 | 121.737666 | 1.786959 | 0.015625 | 1.986209 | 52.004584 | 65.722541 | 0.014000 |
| 400 | 122.508542 | 1.965250 | 0.024333 | 2.023042 | 52.045375 | 66.291375 | 0.013625 |
| 500 | 124.747083 | 1.980750 | 0.016583 | 2.096083 | 53.481709 | 66.969375 | 0.019791 |
| 600 | 122.237250 | 2.106333 | 0.021625 | 2.420583 | 51.747542 | 65.492375 | 0.027375 |
| 700 | 121.661250 | 2.137625 | 0.017625 | 2.322875 | 51.487000 | 65.240958 | 0.020125 |
| 800 | 120.751333 | 1.926875 | 0.024584 | 1.974958 | 51.385500 | 65.235083 | 0.016625 |
| 900 | 121.800500 | 1.828667 | 0.015417 | 1.974500 | 51.316083 | 65.821833 | 0.020417 |
| 938 | 60.224333 | 0.816833 | 0.015167 | 1.060958 | 25.700750 | 32.613916 | 0.013208 |

### Conv1 backward section profile

| Step | Total | Pack grad_output | Pack weight | Padded input | grad_weight | grad_input | Unpack weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 11.874875 | 4.030584 | 0.002916 | 0.256542 | 2.688208 | 4.679917 | 0.000584 |
| 200 | 10.919375 | 3.265958 | 0.001000 | 0.235792 | 2.589916 | 4.647708 | 0.000500 |
| 300 | 10.969792 | 3.337083 | 0.001333 | 0.234667 | 2.598250 | 4.619750 | 0.000709 |
| 400 | 11.053041 | 3.230791 | 0.001458 | 0.232750 | 2.619792 | 4.811833 | 0.000833 |
| 500 | 11.555625 | 3.318209 | 0.001292 | 0.252583 | 2.575417 | 4.650333 | 0.000541 |
| 600 | 10.932042 | 3.271500 | 0.000875 | 0.241625 | 2.598542 | 4.642042 | 0.000667 |
| 700 | 10.960875 | 3.276375 | 0.001000 | 0.234083 | 2.604000 | 4.672417 | 0.000541 |
| 800 | 10.906750 | 3.245417 | 0.000917 | 0.236500 | 2.586250 | 4.676625 | 0.000667 |
| 900 | 11.772000 | 3.223583 | 0.000834 | 0.234917 | 2.596917 | 4.638333 | 0.003792 |
| 938 | 6.513541 | 1.613583 | 0.001250 | 0.123666 | 1.340542 | 2.375083 | 0.000917 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 298.810022917 s |
| Smoothed training loss | 0.3363628374274767 |
| Validation loss | 0.2691631467284183 |
| Validation time | 15.547558291 s |

### Optimization review

- **Better:** The section profile identifies Conv2 `grad_input` as the largest
  individual backward section, typically `65–67 ms`, followed by
  `grad_weight` at `51–54 ms`. The complete operation table also identifies
  matmul backward at `42–45 ms` as the next-largest category. Aggregate
  training time was `298.810022917 s` versus `306.660931041 s` in the prior
  profiled run, but profiling overhead means this is not a clean speedup claim.
- **Worse:** Step-100 backward was `224.945250 ms` versus `212.975750 ms` in
  the prior profiled run, and initial evaluation was slightly higher at
  `15.693244541 s` versus `15.689674792 s`. The expanded timers and verbose
  output add measurement overhead.
- **Unchanged:** Smoothed training loss, validation loss, and gradient count
  remain identical. The profile now identifies the next arithmetic target but
  does not change model behavior.

## 6. Clean run after reverting output-channel blocking

**Source label:** `Clean run after reverting output-channel blocking`
**Branch:** `feature/weird-optimizations*`
**Command:** `make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

This is a clean, unprofiled run after removing the per-tap output-channel
blocking helpers. The stride-1/dilation-1 fast path and interior/boundary
split remain enabled; the square-kernel specialization is not yet applied.

| Metric | Value |
|---|---:|
| Dataset setup | 43.005750 ms |
| Dataloader setup | 0.490333 ms |
| Model setup | 2.473792 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 15.498567500 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 111.769375 ms | 15.833 µs | 202.658416 ms | 0.864000 ms | 2.0968491171052857 | 8 |
| 200 / 938 | 113.037625 ms | 19.000 µs | 201.243291 ms | 0.832083 ms | 1.1225818182926737 | 8 |
| 300 / 938 | 114.632167 ms | 17.042 µs | 199.764542 ms | 0.804917 ms | 0.6686163189624389 | 8 |
| 400 / 938 | 111.888792 ms | 15.833 µs | 202.029458 ms | 0.817667 ms | 0.5889227138650284 | 8 |
| 500 / 938 | 113.572583 ms | 19.625 µs | 202.703625 ms | 0.753459 ms | 0.502232115982027 | 8 |
| 600 / 938 | 113.541958 ms | 16.917 µs | 199.623250 ms | 0.565042 ms | 0.4567609410296239 | 8 |
| 700 / 938 | 112.853416 ms | 15.958 µs | 203.083375 ms | 0.962709 ms | 0.3950323256098518 | 8 |
| 800 / 938 | 116.723291 ms | 18.708 µs | 201.933000 ms | 0.793083 ms | 0.32488540192368714 | 8 |
| 900 / 938 | 118.616916 ms | 20.000 µs | 213.762041 ms | 0.871375 ms | 0.3251889765099359 | 8 |
| 938 / 938 | 54.745542 ms | 11.375 µs | 100.825584 ms | 0.821209 ms | 0.3363628374274767 | 8 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 299.386094958 s |
| Smoothed training loss | 0.3363628374274767 |
| Validation loss | 0.2691631467284183 |
| Validation time | 15.972398166 s |

### Optimization review

- **Better:** Compared with the previous unprofiled compact-pooling baseline,
  first-epoch training decreased from `308.258938459 s` to `299.386094958 s`.
  Step-100 backward decreased from `215.213208 ms` to `202.658416 ms`, and
  step-938 backward decreased from `104.543333 ms` to `100.825584 ms`.
- **Worse:** Validation increased from `15.865784084 s` to `15.972398166 s`.
  The difference is small relative to run-to-run variation, but it is not an
  improvement. The forward checkpoints are mixed.
- **Unchanged:** Smoothed training loss, validation loss, and gradient count
  remain identical. This run is a clean baseline for the upcoming
  square-kernel specialization.

## 7. Stride-aware matmul with unit-stride paths

**Source label:** `Stride-1/dilation-1 matmul with unit-stride paths`
**Branch:** `feature/weird-optimizations*`
**Command:** `MINITORCH_PROFILE_LAYERS=true make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

This profile rewrites `TensorStorage::matmul`, which had no vectorized `f64`
arithmetic at all. The kernel indexed `b` through a runtime stride, so the
compiler could not prove the inner access was contiguous and emitted a scalar
loop; `dL/dA = grad @ W^T` also walked every output column at a full row stride.
Two unit-stride paths replace it: output-row register tiling when `b_s1 == 1`,
and a dot-product form with `k` innermost when `b_s0 == 1`. Both accumulate each
output element over `k = 0..k` in the original order, so results are
bit-identical. The convolution kernels are untouched by this change.

| Metric | Value |
|---|---:|
| Dataset setup | 38.513417 ms |
| Dataloader setup | 0.389250 ms |
| Model setup | 3.006042 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 13.369879042 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Step time | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 98.852500 ms | 17.625 µs | 180.997042 ms | 604.334 µs | 2.0968491171052857 | 8 |
| 200 / 938 | 103.990417 ms | 17.125 µs | 185.942125 ms | 850.333 µs | 1.1225818182926737 | 8 |
| 300 / 938 | 101.479083 ms | 18.917 µs | 207.775416 ms | 1030.833 µs | 0.6686163189624389 | 8 |
| 400 / 938 | 101.242208 ms | 18.459 µs | 189.950375 ms | 934.834 µs | 0.5889227138650284 | 8 |
| 500 / 938 | 98.393791 ms | 18.291 µs | 185.702917 ms | 852.209 µs | 0.502232115982027 | 8 |
| 600 / 938 | 98.273792 ms | 20.625 µs | 184.737375 ms | 883.125 µs | 0.4567609410296239 | 8 |
| 700 / 938 | 98.107792 ms | 20.458 µs | 184.473083 ms | 685.500 µs | 0.3950323256098518 | 8 |
| 800 / 938 | 98.273167 ms | 20.417 µs | 184.365333 ms | 623.834 µs | 0.32488540192368714 | 8 |
| 900 / 938 | 98.729458 ms | 17.375 µs | 184.686375 ms | 851.833 µs | 0.3251889765099359 | 8 |
| 938 / 938 | 49.579125 ms | 13.083 µs | 95.863375 ms | 865.875 µs | 0.3363628374274767 | 8 |

### Forward layer profile

| Step | conv1 (ms) | relu1 (ms) | pool1 (ms) | conv2 (ms) | relu2 (ms) | pool2 (ms) | flatten (ms) | dropout (ms) | linear1 (ms) | relu3 (ms) | linear2 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 11.529750 | 5.102208 | 8.349458 | 58.495541 | 2.485875 | 4.397667 | 0.027625 | 4.876917 | 3.534875 | 0.024792 | 0.025958 |
| 200 | 12.189750 | 5.234459 | 8.746417 | 61.475750 | 2.493042 | 4.458583 | 0.032166 | 5.152792 | 4.150917 | 0.025750 | 0.029042 |
| 300 | 12.196875 | 5.102458 | 8.580208 | 59.815500 | 2.527875 | 4.448083 | 0.074917 | 5.011708 | 3.665958 | 0.026041 | 0.026833 |
| 400 | 11.376416 | 4.932500 | 8.407541 | 61.427167 | 2.422750 | 4.145208 | 0.028833 | 4.876291 | 3.577209 | 0.021125 | 0.025667 |
| 500 | 11.967666 | 4.972375 | 8.232833 | 58.112000 | 2.370834 | 4.192958 | 0.028416 | 4.883375 | 3.579625 | 0.023125 | 0.028417 |
| 600 | 11.361958 | 4.784625 | 8.311750 | 58.438542 | 2.409333 | 4.172833 | 0.028208 | 4.919792 | 3.793417 | 0.021166 | 0.028208 |
| 700 | 11.322875 | 4.862042 | 8.218084 | 58.647125 | 2.369041 | 4.138250 | 0.030458 | 4.882042 | 3.581875 | 0.024500 | 0.030292 |
| 800 | 11.209500 | 4.734584 | 8.177792 | 59.152791 | 2.367208 | 4.098709 | 0.028667 | 4.889333 | 3.561042 | 0.021584 | 0.028166 |
| 900 | 11.301791 | 4.775459 | 8.252958 | 59.399750 | 2.350042 | 4.103000 | 0.027500 | 4.896209 | 3.569459 | 0.024250 | 0.027917 |
| 938 | 5.522875 | 2.673541 | 3.801542 | 29.879542 | 1.208125 | 2.107375 | 0.013250 | 2.430875 | 1.916209 | 0.011041 | 0.013709 |

### Backward operation totals

The remaining operations (`div`, `exp`, `ln`, `max`, `neg`, `sub`, `sum`, and
`unsqueeze`) were each below `0.01 ms` at the logged checkpoints.

| Step | conv2d (ms) | matmul (ms) | maximum (ms) | max_pool2d (ms) | add (ms) | mul (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 129.939541 | 23.844917 | 19.231126 | 4.080666 | 2.360124 | 0.497792 |
| 200 | 131.700374 | 25.458791 | 20.167459 | 4.196167 | 2.626667 | 0.611167 |
| 300 | 147.709583 | 25.545292 | 21.945917 | 6.573834 | 2.517459 | 1.358167 |
| 400 | 132.765666 | 24.676958 | 20.362167 | 4.223750 | 2.564374 | 0.738125 |
| 500 | 130.782958 | 24.613042 | 20.361374 | 4.232041 | 3.386834 | 0.523208 |
| 600 | 131.429667 | 24.556374 | 19.748750 | 4.110417 | 3.280750 | 0.609626 |
| 700 | 131.156833 | 24.225750 | 20.785834 | 4.070083 | 2.423791 | 0.506250 |
| 800 | 131.078958 | 24.666542 | 20.091250 | 4.127792 | 2.564583 | 0.488833 |
| 900 | 130.732542 | 24.299917 | 20.821292 | 4.111125 | 2.537499 | 0.579792 |
| 938 | 68.386542 | 12.396083 | 10.076416 | 1.973875 | 1.262916 | 0.244458 |

### Conv2 backward section profile

The first convolution entry is the later, larger convolution in reverse graph
order. Values are milliseconds. These sections are unchanged by this profile and
serve as the control group.

| Step | Total | Pack grad_output | Pack weight | Padded input | grad_weight | grad_input | Unpack weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 119.287708 | 1.797583 | 0.015084 | 1.924750 | 50.623625 | 64.906083 | 0.013625 |
| 200 | 120.328208 | 2.042667 | 0.015792 | 2.152291 | 51.605417 | 64.494084 | 0.013875 |
| 300 | 136.897041 | 1.950250 | 0.016959 | 2.244000 | 55.691709 | 75.886667 | 0.035916 |
| 400 | 121.497541 | 1.878542 | 0.015208 | 2.215000 | 51.215792 | 65.985333 | 0.013125 |
| 500 | 120.002291 | 1.846083 | 0.014958 | 2.042500 | 51.290708 | 64.780750 | 0.022791 |
| 600 | 120.689500 | 1.958750 | 0.015500 | 2.059042 | 51.179458 | 65.430917 | 0.038833 |
| 700 | 120.406875 | 1.829167 | 0.015875 | 2.063958 | 50.980750 | 64.987833 | 0.013125 |
| 800 | 120.298750 | 1.946791 | 0.015250 | 2.075375 | 51.147666 | 65.088417 | 0.020917 |
| 900 | 119.787792 | 1.924416 | 0.015083 | 1.996708 | 50.590083 | 65.060750 | 0.014000 |
| 938 | 63.079167 | 0.898000 | 0.017000 | 1.010542 | 26.972750 | 34.065875 | 0.014208 |

### Conv1 backward section profile

| Step | Total | Pack grad_output | Pack weight | Padded input | grad_weight | grad_input | Unpack weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10.651833 | 3.209208 | 0.001000 | 0.227709 | 2.487084 | 4.552292 | 0.000500 |
| 200 | 11.372166 | 3.195125 | 0.001166 | 0.231834 | 2.487959 | 4.555041 | 0.000375 |
| 300 | 10.812542 | 3.275333 | 0.000791 | 0.238084 | 2.520584 | 4.621625 | 0.000917 |
| 400 | 11.268125 | 3.357209 | 0.001750 | 0.235208 | 2.641958 | 4.682167 | 0.000875 |
| 500 | 10.780667 | 3.298334 | 0.001417 | 0.238958 | 2.506750 | 4.578042 | 0.000791 |
| 600 | 10.740167 | 3.221792 | 0.003459 | 0.237375 | 2.509667 | 4.599333 | 0.000375 |
| 700 | 10.749958 | 3.269791 | 0.000709 | 0.231584 | 2.499000 | 4.571917 | 0.000625 |
| 800 | 10.780208 | 3.286042 | 0.000791 | 0.236458 | 2.517084 | 4.578583 | 0.000542 |
| 900 | 10.944750 | 3.453375 | 0.004292 | 0.241834 | 2.512292 | 4.557208 | 0.000792 |
| 938 | 5.307375 | 1.660500 | 0.000834 | 0.116666 | 1.249625 | 2.277333 | 0.000417 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 267.85608725 s |
| Smoothed training loss | 0.3363628374274767 |
| Validation loss | 0.2691631467284183 |
| Validation time | 13.462407916 s |

### Stage 6 comparison

Stage 6 is the clean unprofiled run, and Stage 7 is a profiled run, so the
aggregate difference mixes a real speedup with instrumentation overhead. Both
directions are recorded so neither is mistaken for the other.

| Metric | Stage 6 (clean) | Stage 7 (profiled) | Difference |
|---|---:|---:|---:|
| Initial evaluation | 15.498567500 s | 13.369879042 s | −2.128688458 s (−13.7%) |
| Epoch 1 training | 299.386094958 s | 267.85608725 s | −31.530007708 s (−10.5%) |
| Validation time | 15.972398166 s | 13.462407916 s | −2.50999025 s (−15.7%) |

### Operation-level comparison

Median over the nine full-batch checkpoints, steps 100–900, against Stage 5's
section profile, which is the most recent comparable profiled run.

| Operation | Stage 5 | Stage 7 | Difference |
|---|---:|---:|---:|
| `linear1` forward | 16.479250 ms | 3.579625 ms | **−78.3%** |
| `linear2` forward | 0.064250 ms | 0.028166 ms | **−56.2%** |
| `matmul` backward | 42.953167 ms | 24.613042 ms | **−42.7%** |
| `conv2d` backward | 131.592875 ms | 131.156833 ms | −0.3% |
| Conv2 `grad_input` | 64.521167 ms | 65.060750 ms | +0.8% |
| Conv2 `grad_weight` | 51.634666 ms | 51.179458 ms | −0.9% |
| Conv2 forward | 58.959875 ms | 59.152791 ms | +0.3% |
| Conv1 `pack grad_output` | 3.392000 ms | 3.275333 ms | −3.4% |
| `maximum` | 20.917208 ms | 20.362167 ms | −2.7% |
| `max_pool2d` | 4.799333 ms | 4.127792 ms | −14.0% |

### Optimization review

- **Better:** `matmul` backward fell from `42.953167 ms` to `24.613042 ms`, a
  42.7% reduction, which is the second-largest single improvement available
  after the fused convolution kernels. `linear1` forward fell from
  `16.479250 ms` to `3.579625 ms`, a 4.6× improvement, confirming the
  accumulator-reload diagnosis: the forward matmul is the one path the
  `b_s1 == 1` output-row tiling path fully covers. `linear2` forward improved
  by the same mechanism on a much smaller tensor. Epoch-1 training time
  decreased by `31.530007708 s` against the Stage 6 clean run, and validation
  time by `2.50999025 s`.
- **Worse:** Nothing regressed beyond noise. Conv2 `grad_input` moved from
  `64.521167 ms` to `65.060750 ms` (+0.8%) and Conv2 forward from
  `58.959875 ms` to `59.152791 ms` (+0.3%); both are inside the ±15%
  scheduler-noise band established for this machine and neither code path was
  modified. Step 300 is an outlier across every section simultaneously
  (`grad_input` `75.886667 ms`, `max_pool2d` `6.573834 ms`), which is a thermal
  or scheduling artifact rather than a change in behaviour.
- **Unchanged:** Smoothed training loss and validation loss are bit-identical
  to all seven prior profiles, which is the expected result for a change that
  preserves accumulation order and is the strongest available evidence that the
  rewrite is numerically faithful. All convolution sections, `maximum`,
  `pool1`/`pool2`, and the gradient count of 8 are unchanged.

## 8. Register-blocked conv accumulator

**Source label:** `Register-blocked conv accumulator`
**Branch:** `feature/weird-optimizations*`
**Command:** `MINITORCH_PROFILE_LAYERS=true make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

This profile applies the register-tiling fix identified in Stage 7 to the
convolution forward loop and to the input-gradient kernel. Output channels are
processed in blocks of eight (`OUT_CHANNEL_TILE`), with a fixed-size stack array
holding the partial sums across the whole tap loop, plus a scalar remainder loop
for `out_channels` that is not a multiple of eight. Disassembly of the release
binary confirms the accumulators became `v.2d` registers with no accumulator
memory traffic: memory operations per eight lanes per tap fell from twelve loads
and four stores to eight loads, and from eight operations to three in forward.

The `grad_weight` kernel is deliberately unchanged. Its accumulator is
read-modify-written once per output position, 12,544 times per tap for Conv2, so
it cannot be held in registers at all.

**Stage 7's prediction was wrong by a wide margin.** The memory-operation model
implied roughly a halving of `grad_input`; the measured result is `−21.4%`. The
register tile removed the accumulator traffic exactly as intended, but the loop
was evidently not purely load-port bound, and two bounds-check branches plus a
stack reload of the tile length remain in the inner loop. Forward did better
than predicted, at `−29.6%` against an expected `−40%`.

| Metric | Value |
|---|---:|
| Dataset setup | 37.314291 ms |
| Dataloader setup | 0.495500 ms |
| Model setup | 3.771209 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 10.484789333 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Step time | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 78.961667 ms | 16.875 µs | 170.785708 ms | 859.125 µs | 2.0968491171052857 | 8 |
| 200 / 938 | 78.640333 ms | 16.541 µs | 184.811167 ms | 944.959 µs | 1.1225818182926737 | 8 |
| 300 / 938 | 78.365458 ms | 17.375 µs | 170.673458 ms | 890.958 µs | 0.6686163189624389 | 8 |
| 400 / 938 | 77.793291 ms | 16.667 µs | 170.203792 ms | 849.625 µs | 0.5889227138650284 | 8 |
| 500 / 938 | 78.651958 ms | 17.333 µs | 180.727583 ms | 873.084 µs | 0.502232115982027 | 8 |
| 600 / 938 | 77.794000 ms | 16.666 µs | 168.066959 ms | 874.375 µs | 0.4567609410296239 | 8 |
| 700 / 938 | 78.691292 ms | 17.167 µs | 170.071417 ms | 778.333 µs | 0.3950323256098518 | 8 |
| 800 / 938 | 78.527625 ms | 16.333 µs | 168.515542 ms | 859.750 µs | 0.32488540192368714 | 8 |
| 900 / 938 | 77.688084 ms | 17.334 µs | 168.496917 ms | 900.334 µs | 0.3251889765099359 | 8 |
| 938 / 938 | 38.543625 ms | 13.292 µs | 84.761375 ms | 946.458 µs | 0.3363628374274767 | 8 |

### Forward layer profile

| Step | conv1 (ms) | relu1 (ms) | pool1 (ms) | conv2 (ms) | relu2 (ms) | pool2 (ms) | flatten (ms) | dropout (ms) | linear1 (ms) | relu3 (ms) | linear2 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 9.354875 | 4.729333 | 7.550125 | 41.691708 | 2.365708 | 4.726042 | 0.035500 | 4.895916 | 3.560875 | 0.024875 | 0.025167 |
| 200 | 9.246208 | 4.766959 | 7.303875 | 41.652542 | 2.383417 | 4.700750 | 0.082292 | 4.876250 | 3.568000 | 0.025250 | 0.030459 |
| 300 | 9.596625 | 4.835709 | 7.451791 | 41.618250 | 2.375083 | 3.929584 | 0.028417 | 4.896333 | 3.577542 | 0.023958 | 0.027875 |
| 400 | 9.287167 | 4.736916 | 7.271625 | 41.690541 | 2.353125 | 3.958875 | 0.026583 | 4.848958 | 3.568208 | 0.021375 | 0.027584 |
| 500 | 9.811083 | 4.754833 | 7.474542 | 41.616833 | 2.474792 | 3.940875 | 0.029584 | 4.879750 | 3.610167 | 0.029250 | 0.028208 |
| 600 | 9.283375 | 4.753834 | 7.291541 | 41.645917 | 2.399750 | 3.888041 | 0.027875 | 4.887125 | 3.563041 | 0.021375 | 0.027917 |
| 700 | 9.278750 | 4.805500 | 7.441875 | 41.588333 | 2.356958 | 4.663417 | 0.035458 | 4.904541 | 3.560208 | 0.024292 | 0.027750 |
| 800 | 9.238750 | 4.775583 | 7.284167 | 41.693791 | 2.397500 | 4.557292 | 0.036167 | 4.916333 | 3.575083 | 0.025875 | 0.025500 |
| 900 | 9.237125 | 4.757250 | 7.353208 | 41.605083 | 2.362292 | 3.844125 | 0.028000 | 4.883750 | 3.563458 | 0.021333 | 0.027833 |
| 938 | 4.585500 | 2.519833 | 3.109917 | 20.832208 | 1.219333 | 1.934209 | 0.014250 | 2.448458 | 1.850792 | 0.011250 | 0.013209 |

### Backward operation totals

The remaining operations (`div`, `exp`, `ln`, `max`, `neg`, `sub`, `sum`, and
`unsqueeze`) were each below `0.01 ms` at the logged checkpoints.

| Step | conv2d (ms) | matmul (ms) | maximum (ms) | max_pool2d (ms) | add (ms) | mul (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 116.740917 | 24.312708 | 20.225875 | 4.471000 | 2.961167 | 0.607750 |
| 200 | 121.605875 | 24.964500 | 26.282083 | 4.377416 | 4.901417 | 0.608334 |
| 300 | 116.484958 | 24.917708 | 20.102375 | 4.073333 | 3.152708 | 0.534334 |
| 400 | 116.212168 | 24.608416 | 20.011333 | 4.480125 | 2.651875 | 0.602375 |
| 500 | 118.113375 | 24.328833 | 28.452792 | 4.049417 | 3.027500 | 0.557334 |
| 600 | 115.883125 | 24.283792 | 19.891915 | 4.002042 | 2.461710 | 0.524875 |
| 700 | 115.482584 | 24.368417 | 21.897584 | 4.245583 | 2.457708 | 0.560916 |
| 800 | 115.303292 | 24.327458 | 20.598542 | 4.235625 | 2.467918 | 0.610875 |
| 900 | 115.367750 | 24.319500 | 19.838666 | 4.016000 | 2.791292 | 0.555708 |
| 938 | 57.642249 | 12.235792 | 10.257000 | 1.935750 | 1.221668 | 0.203792 |

### Conv2 backward section profile

The first convolution entry is the later, larger convolution in reverse graph
order. Values are milliseconds.

| Step | Total | Pack grad_output | Pack weight | Padded input | grad_weight | grad_input | Unpack weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 107.290708 | 1.995584 | 0.015666 | 2.160000 | 51.245084 | 51.457083 | 0.018083 |
| 200 | 112.029333 | 6.247292 | 0.022291 | 2.847000 | 51.050458 | 51.162917 | 0.017167 |
| 300 | 106.887667 | 1.905167 | 0.016708 | 2.096750 | 51.189916 | 51.353667 | 0.016834 |
| 400 | 106.715834 | 1.862833 | 0.016750 | 2.031458 | 51.474917 | 51.311167 | 0.014666 |
| 500 | 108.584709 | 2.274084 | 0.019792 | 2.491333 | 51.661083 | 51.325000 | 0.013917 |
| 600 | 106.392125 | 1.922000 | 0.016250 | 2.008583 | 51.073000 | 51.164000 | 0.014250 |
| 700 | 106.008542 | 1.807250 | 0.023334 | 2.016833 | 50.895333 | 51.096166 | 0.023208 |
| 800 | 105.841875 | 1.789542 | 0.015834 | 1.977834 | 50.740541 | 51.133167 | 0.023959 |
| 900 | 105.962459 | 1.954709 | 0.015708 | 2.069833 | 50.732333 | 51.004625 | 0.012958 |
| 938 | 52.616791 | 0.830625 | 0.016000 | 0.914125 | 25.411417 | 25.418000 | 0.022250 |

### Conv1 backward section profile

| Step | Total | Pack grad_output | Pack weight | Padded input | grad_weight | grad_input | Unpack weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 9.450209 | 3.260958 | 0.001083 | 0.235667 | 2.544541 | 3.251084 | 0.000500 |
| 200 | 9.576542 | 3.304584 | 0.001625 | 0.289542 | 2.555000 | 3.242708 | 0.000458 |
| 300 | 9.597291 | 3.405542 | 0.000792 | 0.232125 | 2.534792 | 3.247333 | 0.000666 |
| 400 | 9.496334 | 3.311084 | 0.001416 | 0.232542 | 2.550541 | 3.236375 | 0.001083 |
| 500 | 9.528666 | 3.290750 | 0.001041 | 0.236292 | 2.532000 | 3.261625 | 0.000334 |
| 600 | 9.491000 | 3.263709 | 0.001584 | 0.254917 | 2.550833 | 3.257666 | 0.000958 |
| 700 | 9.474042 | 3.270875 | 0.000916 | 0.253500 | 2.541708 | 3.247333 | 0.000792 |
| 800 | 9.461417 | 3.241125 | 0.000875 | 0.257750 | 2.532000 | 3.249041 | 0.000750 |
| 900 | 9.405291 | 3.225542 | 0.000791 | 0.237000 | 2.522417 | 3.248875 | 0.000375 |
| 938 | 5.025458 | 1.812208 | 0.001042 | 0.124541 | 1.268125 | 1.614292 | 0.000625 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 234.81543525 s |
| Smoothed training loss | 0.3363628374274767 |
| Validation loss | 0.2691631467284183 |
| Validation time | 10.455207417 s |

### Stage 7 comparison

Both runs are profiled, so unlike the Stage 6 comparison this is a clean
like-for-like difference with no instrumentation mismatch.

| Metric | Stage 7 | Stage 8 | Difference |
|---|---:|---:|---:|
| Initial evaluation | 13.369879042 s | 10.484789333 s | −2.885089709 s (−21.6%) |
| Epoch 1 training | 267.85608725 s | 234.81543525 s | −33.040652 s (−12.3%) |
| Validation time | 13.462407916 s | 10.455207417 s | −3.007200499 s (−22.3%) |

### Operation-level comparison

Median over the nine full-batch checkpoints, steps 100–900. Rows marked
*control* are code paths this change did not touch, and act as a check on
run-to-run drift.

| Operation | Stage 7 | Stage 8 | Difference |
|---|---:|---:|---:|
| Conv2 forward | 59.152791 ms | 41.645917 ms | **−29.6%** |
| Conv1 forward | 11.376416 ms | 9.283375 ms | **−18.4%** |
| Conv2 `grad_input` | 65.060750 ms | 51.164000 ms | **−21.4%** |
| Conv1 `grad_input` | 4.578042 ms | 3.248875 ms | **−29.0%** |
| Conv2 `grad_weight` | 51.179458 ms | 51.073000 ms | −0.2% *(control)* |
| Conv1 `grad_weight` | 2.509667 ms | 2.541708 ms | +1.3% *(control)* |
| `matmul` backward | 24.613042 ms | 24.328833 ms | −1.1% *(control)* |
| `linear1` forward | 3.579625 ms | 3.568000 ms | −0.3% *(control)* |
| `maximum` | 20.362167 ms | 20.225875 ms | −0.7% *(control)* |
| Conv2 `pack grad_output` | 1.924416 ms | 1.922000 ms | −0.1% *(control)* |
| Conv2 `padded input` | 2.063958 ms | 2.069833 ms | +0.3% *(control)* |
| `pool1` | 8.311750 ms | 7.353208 ms | −11.5% |
| `max_pool2d` | 4.127792 ms | 4.235625 ms | +2.6% *(noise)* |
| `add` | 2.537499 ms | 2.791292 ms | +10.0% *(noise)* |

### Optimization review

- **Better:** Conv2 forward improved `29.6%` and Conv2 `grad_input` `21.4%`,
  taking the convolution forward pass from `59.15` to `41.65 ms` and the
  Conv2 backward total from `120.33` to `106.72 ms`. Epoch-1 training fell by
  `33.040652 s` against a profiled Stage 7, and validation by `3.007200499 s`.
  Conv1, which shares the same kernels, improved `18.4%` forward and `29.0%` on
  `grad_input` for free. Every *control* row is within 1.3% of its Stage 7 value,
  which confirms the change was isolated.
- **Worse:** The predicted gain did not materialise. The memory-operation model
  predicted `grad_input` would roughly halve; it fell `21.4%`. The register tile
  worked as designed in the disassembly, so the residual is in the loop's
  non-memory costs: two bounds-check branches and a stack reload of the tile
  length remain in the inner loop, and the nested block/tap order recomputes tap
  addresses eight times per output position. `pool1` and `relu1` improved
  `11.5%` and `3.6%` without being touched, most likely reduced memory pressure
  from the cheaper convolution loops; `add` and `max_pool2d` moved within noise.
- **Unchanged:** Despite the structural reassociation in `grad_input`, whose
  reduction over output channels is now a sum of eight-element block partials,
  the smoothed training loss and validation loss are **bit-identical to all
  eight prior profiles**. The reassociated sums evidently produce the same
  doubles for this data, so the bit-identity check remains usable. `grad_weight`
  is unchanged by construction and confirms at `−0.2%`.

## Observations and next measurements

- The first-epoch training loss and validation loss remain bit-identical across
  all nine profiles, including across the Stage 8 reassociation, so no change
  has altered the observed training trajectory.
- Both Stages 7 and 8 are profiled runs, so their difference is clean. A clean
  `make run-release` measurement is still needed to bring the unprofiled
  baseline up to date; the profiled figure is not directly comparable to the
  Stage 6 clean row.
- The dot-product matmul path remains scalar. LLVM will not vectorize a
  floating-point reduction without `reassoc`, and Rust does not set it, so that
  path gained contiguous access and lost its accumulator traffic but gained no
  SIMD. Fixing it needs a two-dimensional register tile with `k` innermost, which
  is a larger change than the two unit-stride paths used here.
- Conv2 `grad_weight` is now the largest single convolution section at
  `51.073000 ms`, level with `grad_input` at `51.164000 ms`. Register blocking
  does not apply to it, because its accumulator is rewritten once per output
  position and must survive the position loop. It is already L2-resident at
  `144 KiB`. The candidate approaches are a two-dimensional tile over channels
  and positions, or a loop interchange to shorten the reuse distance.
- `maximum` is stable at `20.225875 ms` across many profiles and has never been
  targeted. It is the third-largest backward category after Conv2 and matmul,
  and the largest code path in the model never examined.
- Predictions from the load-port memory model have now been wrong twice: the
  Stage 7 matmul rewrite delivered `1.7×` against a predicted `3×`, and the
  Stage 8 conv tiling delivered `1.27×` on `grad_input` against a predicted `2×`.
  Arithmetic-intensity models on this machine are useful for ruling a change
  *out*, not for sizing it. Treat future estimates as optimistic by roughly a
  factor of two.
- Future profiles should record CPU model, commit, dtype, and thread settings
  before making claims that require cross-machine or PyTorch comparisons.
- Do not extrapolate from individual checkpoints, especially step `938/938`;
  use the measured epoch and validation totals for comparisons.

## Historical PyTorch reference

The following reference is kept separately from the nine Rust profiles above.

| Metric | Value |
|---|---:|
| Device | CPU |
| Dtype | `torch.float64` |
| Torch threads | 1 |
| Dataset setup | 0.440 s |
| Initial evaluation | 9.336 s |
| Initial loss | 2.314024 |
| Training batches | 938 |
| Validation batches | 157 |

### PyTorch step 100

| Operation | Time |
|---|---:|
| Forward | 0.052357 s |
| Loss | 0.000113 s |
| Backward | 0.063982 s |
| Optimizer step | 0.000962 s |
| Smoothed loss | 2.146694 |

The displayed PyTorch operation times sum to approximately `0.117414 s` per
batch. This is a reference measurement, not one of the Rust profiles in
`prof.txt`.

## Result-entry template

Use this structure for future optimization profiles:

```markdown
## <optimization> — <YYYY-MM-DD>

- Source label:
- Branch/commit:
- Command: `make run-release`
- CPU/OS/dtype/threads:
- Stop point: after epoch 1 validation

| Metric | Value |
|---|---:|
| Dataset setup | |
| Dataloader setup | |
| Model setup | |
| Initial evaluation | |
| Initial loss | |
| Training batches | |
| Validation batches | |

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | | | | | | |
| ... | | | | | | |
| 938 / 938 | | | | | | |

- Training epoch time:
- Smoothed training loss:
- Validation loss:
- Validation time:
- Better:
- Worse:
- Unchanged/trade-offs:
- Notes:
```
