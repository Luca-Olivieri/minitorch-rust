# MNIST Optimization History

This file records the Rust MNIST `SmallCNN` optimization measurements. The
seven Rust profiles below are transcribed from the recorded optimization runs;
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
- `grads.len() = 8` was reported at every checkpoint in all seven profiles.
- Record CPU model, operating system, exact commit, dtype, and thread settings
  with each profile when making a cross-machine comparison.

## Summary

All seven runs use the same dataset sizes reported by the program: 60,000
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

The seven profiles follow the same measurement boundary: initial evaluation,
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

## Observations and next measurements

- The first-epoch training loss and validation loss remain effectively identical
  across all seven profiles, so the logging, fast-path, and blocking-revert
  changes did not alter the observed training trajectory.
- The clean post-revert run is the current unprofiled baseline. The next
  optimization target remains Conv2 `grad_input`, with matmul backward as the
  next-largest category from the section profile.
- Future profiles should record CPU model, commit, dtype, and thread settings
  before making claims that require cross-machine or PyTorch comparisons.
- Do not extrapolate from individual checkpoints, especially step `938/938`;
  use the measured epoch and validation totals for comparisons.

## Historical PyTorch reference

The following reference is kept separately from the seven Rust profiles above.

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
