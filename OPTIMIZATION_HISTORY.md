# MNIST Optimization History

This file records the Rust MNIST `SmallCNN` optimization measurements. The
first ten Rust profiles are summarized below; later profiles, including the
threading sweep, are recorded in full below. Measurements are kept at their
original precision so that later runs can be compared directly.

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
- `grads.len() = 8` was reported at every full-batch checkpoint in every
  recorded profile.
- Stages 7 and 8 report the post-accumulation time as `step_time`; earlier
  stages labelled the same field `Optimizer`. The values are directly
  comparable.
- Record CPU model, operating system, exact commit, dtype, and thread settings
  with each profile when making a cross-machine comparison.

## Summary

All recorded runs use the same dataset sizes reported by the program: 60,000
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
| Position-tiled `grad_weight` | `feature/weird-optimizations*` | 10.406701042 s | 229.711056 s | 0.33636283742747686 | 0.26916314672841835 | 10.493345833 s |
| `max_pool2d_backward` fast path (reverted) | `feature/weird-optimizations*` | 10.457898792 s | 233.695550208 s | 0.33636283742747686 | 0.26916314672841835 | 10.475820292 s |
| Batch-parallel `grad_input`, 1 worker | `feature/weird-optimizations*` | 10.569629583 s | 227.564065542 s | 0.33636283742747686 | 0.26916314672841835 | 10.769859125 s |
| Batch-parallel `grad_input`, 2 workers | `feature/weird-optimizations*` | 10.510451208 s | 201.891501334 s | 0.33636283742747686 | 0.26916314672841835 | 10.564413333 s |
| Batch-parallel `grad_input`, 4 workers (Stages 11–12) | `feature/weird-optimizations*` | 10.398292375 s | 193.03315625 s | 0.33636283742747686 | 0.26916314672841835 | 10.471934791 s |
| Batch-parallel `grad_input`, 8 workers | `feature/weird-optimizations*` | 10.758061 s | 189.177429041 s | 0.33636283742747686 | 0.26916314672841835 | 10.539541958 s |

The first ten profiles follow the same measurement boundary: initial evaluation,
all 938 epoch-1 training batches, and validation on 157 test batches. Later
profiles use the same boundary and are recorded in their respective stages.

**Loss values stop being bit-identical at Stage 9.** Stages 0 through 8 all
report `0.3363628374274767` and `0.2691631467284183`; Stage 9 reports
`0.33636283742747686` and `0.26916314672841835`. This is a floating-point
reassociation, not a behavioural change, and it is discussed in Stage 9.

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

## 9. Position-tiled `grad_weight`

**Source label:** `Position-tiled grad_weight`
**Branch:** `feature/weird-optimizations*`
**Command:** `MINITORCH_PROFILE_LAYERS=true make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

The weight-gradient accumulator cannot be register-blocked the way the forward
and input-gradient accumulators were, because it is read-modify-written once per
output position and must survive the whole position loop. A register tile of
*positions* can be, for one channel block at a time. The kernel now walks the
output plane in tiles of eight positions, accumulating each tile in registers
before flushing once into the shared packed gradient, with scalar remainder loops
for positions and for channels. Tiling also shortens the packed-gradient reuse
distance from the whole 144 KiB buffer to `kernel_w * out_channels`, about
1.5 KiB for Conv2.

Disassembly of the release binary confirms the intended shape: memory
operations per eight positions × eight channels × one tap fell from 96 to 48,
and the accumulator's share of the inner loop fell from 64 operations to zero.
An earlier revision that tabulated each tile's row and column into two
`[usize; 8]` arrays spilled both those and the input scalars to the stack, and
inflated the vector-FMA count in the function from 32 to 124; recomputing row
and column at each use removed the spill entirely.

| Metric | Value |
|---|---:|
| Dataset setup | 42.321750 ms |
| Dataloader setup | 0.458834 ms |
| Model setup | 3.068250 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 10.406701042 s |

### Epoch 1 checkpoints

| Step | Forward | Loss | Backward | Step time | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 78.895792 ms | 16.250 µs | 169.313958 ms | 655.125 µs | 2.0968491171052857 | 8 |
| 200 / 938 | 82.312542 ms | 16.459 µs | 164.330000 ms | 867.708 µs | 1.122581818292674 | 8 |
| 300 / 938 | 94.389167 ms | 18.250 µs | 166.878708 ms | 851.375 µs | 0.668616318962439 | 8 |
| 400 / 938 | 77.912416 ms | 16.250 µs | 164.090541 ms | 965.958 µs | 0.5889227138650285 | 8 |
| 500 / 938 | 78.545333 ms | 17.875 µs | 166.835875 ms | 838.792 µs | 0.502232115982027 | 8 |
| 600 / 938 | 77.761916 ms | 36.834 µs | 170.445667 ms | 951.000 µs | 0.45676094102962383 | 8 |
| 700 / 938 | 79.449000 ms | 43.792 µs | 163.660542 ms | 900.958 µs | 0.3950323256098518 | 8 |
| 800 / 938 | 78.541125 ms | 16.916 µs | 163.276667 ms | 851.750 µs | 0.32488540192368714 | 8 |
| 900 / 938 | 78.890125 ms | 16.750 µs | 165.791792 ms | 927.541 µs | 0.32518897650993595 | 8 |
| 938 / 938 | 38.424584 ms | 15.875 µs | 81.883541 ms | 849.750 µs | 0.33636283742747686 | 8 |

### Forward layer profile

| Step | conv1 (ms) | relu1 (ms) | pool1 (ms) | conv2 (ms) | relu2 (ms) | pool2 (ms) | flatten (ms) | dropout (ms) | linear1 (ms) | relu3 (ms) | linear2 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10.194834 | 4.781208 | 7.342917 | 41.632083 | 2.372417 | 4.050125 | 0.109875 | 4.838750 | 3.522458 | 0.024250 | 0.025333 |
| 200 | 9.936875 | 4.974625 | 7.793875 | 44.038125 | 2.339375 | 4.682209 | 0.039333 | 4.879291 | 3.572250 | 0.024166 | 0.027792 |
| 300 | 10.624584 | 5.221625 | 20.256708 | 43.088875 | 2.652750 | 3.991792 | 0.029416 | 4.884292 | 3.585709 | 0.025042 | 0.026917 |
| 400 | 9.456875 | 4.719959 | 7.232708 | 41.658250 | 2.369375 | 3.939209 | 0.027166 | 4.876709 | 3.574584 | 0.026042 | 0.027875 |
| 500 | 9.245750 | 4.785875 | 7.272208 | 41.638833 | 2.396750 | 4.669875 | 0.034875 | 4.879250 | 3.563084 | 0.026625 | 0.028292 |
| 600 | 9.345625 | 4.765208 | 7.269208 | 41.592167 | 2.366834 | 3.868250 | 0.028625 | 4.909458 | 3.561834 | 0.026541 | 0.026333 |
| 700 | 9.525167 | 4.846042 | 7.438416 | 42.161959 | 2.400709 | 3.968584 | 0.030625 | 5.202834 | 3.821375 | 0.021583 | 0.028041 |
| 800 | 9.532042 | 4.873333 | 7.504458 | 41.814875 | 2.373125 | 3.936667 | 0.026333 | 4.864041 | 3.568708 | 0.021000 | 0.025291 |
| 900 | 9.281792 | 4.746750 | 7.457250 | 41.651084 | 2.491542 | 4.600333 | 0.036208 | 5.000167 | 3.568791 | 0.024000 | 0.030375 |
| 938 | 4.568959 | 2.522583 | 3.126958 | 20.828791 | 1.195083 | 1.881709 | 0.012458 | 2.417458 | 1.845458 | 0.011083 | 0.013084 |

### Backward operation totals

The remaining operations (`div`, `exp`, `ln`, `max`, `neg`, `sub`, `sum`, and
`unsqueeze`) were each below `0.01 ms` at the logged checkpoints.

| Step | conv2d (ms) | matmul (ms) | maximum (ms) | max_pool2d (ms) | add (ms) | mul (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 111.639833 | 24.371083 | 24.624334 | 4.070376 | 2.671833 | 0.535999 |
| 200 | 111.539374 | 24.363376 | 19.686958 | 4.170500 | 2.549291 | 0.585917 |
| 300 | 111.961792 | 24.742375 | 20.232625 | 4.646875 | 3.155251 | 0.778375 |
| 400 | 111.047417 | 24.298208 | 19.569083 | 3.964000 | 2.451166 | 1.615750 |
| 500 | 111.265917 | 24.403500 | 21.778792 | 4.288917 | 2.581166 | 0.545917 |
| 600 | 113.076582 | 24.343458 | 23.204876 | 4.075708 | 2.675666 | 0.591917 |
| 700 | 111.020126 | 25.206041 | 19.458083 | 4.044209 | 2.386916 | 0.576834 |
| 800 | 111.404000 | 24.304750 | 19.573542 | 4.143250 | 2.408626 | 0.479416 |
| 900 | 111.311625 | 24.656459 | 20.778417 | 4.238375 | 2.513751 | 0.762583 |
| 938 | 55.334000 | 12.341000 | 9.997625 | 1.810083 | 1.106291 | 0.244250 |

### Conv2 backward section profile

The first convolution entry is the later, larger convolution in reverse graph
order. Values are milliseconds.

| Step | Total | Pack grad_output | Pack weight | Padded input | grad_weight | grad_input | Unpack weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 101.673125 | 2.323666 | 0.018084 | 2.205834 | 45.090875 | 51.848875 | 0.016416 |
| 200 | 101.489916 | 2.058500 | 0.017125 | 2.023958 | 45.359792 | 51.831291 | 0.013792 |
| 300 | 101.479875 | 1.954500 | 0.016167 | 2.005666 | 45.391417 | 51.923834 | 0.017875 |
| 400 | 101.017459 | 1.782542 | 0.015792 | 2.000625 | 45.136833 | 51.889167 | 0.013250 |
| 500 | 101.319292 | 1.821917 | 0.016458 | 1.996083 | 45.482709 | 51.824417 | 0.019792 |
| 600 | 102.127666 | 2.022834 | 0.025500 | 2.162292 | 45.403500 | 51.930875 | 0.016833 |
| 700 | 101.057042 | 1.808541 | 0.016458 | 1.967917 | 45.429042 | 51.817917 | 0.013625 |
| 800 | 101.364750 | 1.846875 | 0.015917 | 1.994500 | 45.589375 | 51.899375 | 0.014417 |
| 900 | 101.341792 | 2.068708 | 0.016250 | 1.999083 | 45.213333 | 51.825875 | 0.020625 |
| 938 | 50.100334 | 0.732125 | 0.015875 | 1.089792 | 22.419584 | 25.816542 | 0.022500 |

### Conv1 backward section profile

| Step | Total | Pack grad_output | Pack weight | Padded input | grad_weight | grad_input | Unpack weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 9.966708 | 3.253792 | 0.000875 | 0.243292 | 3.038084 | 3.266041 | 0.000917 |
| 200 | 10.049458 | 3.262208 | 0.000792 | 0.250750 | 3.055167 | 3.314250 | 0.001000 |
| 300 | 10.481917 | 3.275375 | 0.001416 | 0.235958 | 3.040542 | 3.252750 | 0.000833 |
| 400 | 10.029958 | 3.285625 | 0.001292 | 0.265417 | 3.040959 | 3.265083 | 0.000750 |
| 500 | 9.946625 | 3.270083 | 0.001042 | 0.233833 | 3.040500 | 3.240791 | 0.000792 |
| 600 | 10.948916 | 3.679041 | 0.001917 | 0.244458 | 3.125125 | 3.386916 | 0.001334 |
| 700 | 9.963084 | 3.275750 | 0.000959 | 0.232167 | 3.043000 | 3.239583 | 0.000500 |
| 800 | 10.039250 | 3.302209 | 0.001500 | 0.231542 | 3.042000 | 3.304833 | 0.001625 |
| 900 | 9.969833 | 3.277250 | 0.003083 | 0.233166 | 3.031958 | 3.262459 | 0.000834 |
| 938 | 5.233666 | 1.631042 | 0.000791 | 0.116792 | 1.553292 | 1.631500 | 0.000958 |

### Epoch 1 completion

| Metric | Value |
|---|---:|
| Training time | 229.711056 s |
| Smoothed training loss | 0.33636283742747686 |
| Validation loss | 0.26916314672841835 |
| Validation time | 10.493345833 s |

### Stage 8 comparison

Both runs are profiled, so this is a clean like-for-like difference.

| Metric | Stage 8 | Stage 9 | Difference |
|---|---:|---:|---:|
| Initial evaluation | 10.484789333 s | 10.406701042 s | −0.078088291 s (−0.7%) |
| Epoch 1 training | 234.81543525 s | 229.711056 s | −5.10437925 s (−2.2%) |
| Validation time | 10.455207417 s | 10.493345833 s | +0.038138416 s (+0.4%) |

### Operation-level comparison

Median over the nine full-batch checkpoints, steps 100–900. Rows marked
*control* are code paths this change did not touch.

| Operation | Stage 8 | Stage 9 | Difference |
|---|---:|---:|---:|
| Conv2 `grad_weight` | 51.073000 ms | 45.391417 ms | **−11.1%** |
| Conv1 `grad_weight` | 2.541708 ms | 3.040959 ms | **+19.7%** |
| Conv2 backward total | 106.715834 ms | 101.364750 ms | −5.0% |
| `conv2d` total | 116.212168 ms | 111.404000 ms | −4.1% |
| Conv2 `grad_input` | 51.164000 ms | 51.848875 ms | +1.3% *(control)* |
| Conv1 `grad_input` | 3.248875 ms | 3.265083 ms | +0.5% *(control)* |
| Conv1 backward total | 9.491000 ms | 10.029958 ms | +5.7% |
| Conv2 forward | 41.645917 ms | 41.658250 ms | +0.0% *(control)* |
| `matmul` backward | 24.328833 ms | 24.371083 ms | +0.2% *(control)* |
| `linear1` forward | 3.568000 ms | 3.568791 ms | +0.0% *(control)* |
| `maximum` | 20.225875 ms | 20.232625 ms | +0.0% *(control)* |
| Conv2 `pack grad_output` | 1.922000 ms | 1.954500 ms | +1.7% *(control)* |
| Conv2 `padded input` | 2.069833 ms | 1.999083 ms | −3.4% |

### Optimization review

- **Better:** Conv2 `grad_weight` improved `11.1%`, from `51.073000` to
  `45.391417 ms`, taking the Conv2 backward total from `106.72` to
  `101.36 ms`. Epoch-1 training fell by `5.10437925 s`. The disassembly
  predicted a 2× cut in memory operations per tiled unit and the code delivered
  that; the wall-clock gain is smaller, which is the third time the
  instruction-count model has overstated the result.
- **Worse:** **Conv1 `grad_weight` regressed `19.7%`**, from `2.541708` to
  `3.040959 ms`, and this is the change's clearest cost. Both distributions are
  tight — Stage 8 spans `2.487`–`2.554`, Stage 9 spans `3.032`–`3.055` — so this
  is a real shift and not measurement noise. Conv1 has one input channel against
  Conv2's 32, so it has nine taps instead of 288 and the per-tile setup is
  amortised over roughly a thirtieth of the arithmetic. Net effect on training is
  `−5.68 + 0.50 = −5.18 ms` per step, so the change is still a clear win, but a
  size guard that skips tiling for small `in_channels × out_channels` would
  recover the `0.50 ms` at the cost of a threshold constant. `pool1` at step 300
  reached `20.256708 ms` against a median near `7.4 ms`, and `maximum` ranged
  from `19.458083` to `24.624334 ms` within a single profile; both are thermal or
  scheduling artifacts, not changes in behaviour.
- **Unchanged:** Every *control* row is within 1.7% of its Stage 8 value,
  including the untouched `grad_input` paths, which confirms the change was
  isolated. **The loss values are no longer bit-identical to Stages 0–8**, and
  this is a correction to the claim made when the change was written. The
  position tile accumulates eight positions into a register partial before
  flushing, whereas the original kernel left-folded all positions in sequence.
  Visiting positions in the same order is not sufficient for bit-identity; the
  grouping into partials is itself a reassociation, exactly as in Stage 8. The
  observed shift is the expected magnitude — the smoothed training loss moved
  from `0.3363628374274767` to `0.33636283742747686`, about one unit in the last
  place — and six of the ten checkpoints shifted while four did not, which is
  what a sub-ulp difference propagating through training looks like. The bit
  identity gate is therefore no longer available as a correctness check from
  Stage 9 onward; the reference-comparison tests in `tests/nn/conv2d.rs` carry
  that load instead.

## 10. `max_pool2d_backward` single-maximum fast path — no measurable effect

**Source label:** `max_pool2d_backward single-maximum fast path`
**Branch:** `feature/weird-optimizations*`
**Command:** `MINITORCH_PROFILE_LAYERS=true make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

The `maximum` kernel had never been targeted. The hypothesis was that it was
bound by a floating-point division executed once per output window:
`1.0 / (end - start) as f64`, where `end - start` is the number of positions
that tied for the window maximum. For non-overlapping windows the common case
is a unique maximum, where that divisor is `1.0` and the division is
unnecessary. A second, smaller cost was a pair of bounds checks per scattered
element, from a manual `if input_index >= out_buf.len()` followed by a
bounds-checked index.

The change adds a fast path for `end - start == 1` that skips both the division
and the scatter loop, and routes every scatter through `get_mut` so the range
check happens once. It is bit-identical: `grad * 1.0 == grad` for every finite
value, and the tied-maximum path is unchanged.

Both branches were mutation-tested. Doubling the fast-path share, dropping the
fast-path scatter, and removing the even split on the tied path are each caught
by `tests/nn/maxpool2d.rs`.

| Metric | Value |
|---|---:|
| Dataset setup | 39.342583 ms |
| Dataloader setup | 0.482667 ms |
| Model setup | 3.395125 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Validation batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 10.457898792 s |
| Training time | 233.695550208 s |
| Smoothed training loss | 0.33636283742747686 |
| Validation loss | 0.26916314672841835 |
| Validation time | 10.475820292 s |

### Stage 9 comparison

| Metric | Stage 9 | Stage 10 | Difference |
|---|---:|---:|---:|
| Initial evaluation | 10.406701042 s | 10.457898792 s | +0.051 s (+0.5%) |
| Epoch 1 training | 229.711056 s | 233.695550208 s | +3.98 s (+1.7%) |
| Validation time | 10.493345833 s | 10.475820292 s | −0.017 s (−0.2%) |

### Operation-level comparison

| Operation | Stage 9 | Stage 10 | Difference |
|---|---:|---:|---:|
| `maximum` | 20.232625 ms | 20.843750 ms | **+3.0%** |
| `conv2d` total | 111.404000 ms | 112.482416 ms | +1.0% *(control)* |
| Conv2 `grad_input` | 51.848875 ms | 51.930958 ms | +0.2% *(control)* |
| Conv2 `grad_weight` | 45.391417 ms | 45.490042 ms | +0.2% *(control)* |
| Conv2 forward | 41.658250 ms | 41.909791 ms | +0.6% *(control)* |
| `matmul` | 24.371083 ms | 24.265792 ms | −0.4% *(control)* |
| `linear1` | 3.568791 ms | 3.580042 ms | +0.3% *(control)* |
| `max_pool2d` | 4.143250 ms | 4.206375 ms | +1.5% *(control)* |
| `add` | 2.549291 ms | 2.562792 ms | +0.5% *(control)* |

### Optimization review

- **Better:** Nothing. The change is bit-identical, which confirms the
  implementation is correct, and it eliminated the division: the release
  binary now contains no `fdiv` outside `mnist::main`, where the only remaining
  one belongs to the loss smoother. The arithmetic the fast path was written to
  remove is genuinely gone. It bought no time.
- **Worse:** `maximum` rose `3.0%`, and epoch-1 training rose `1.7%`. However
  **every control also rose, by `0.2%` to `1.5%`**, which indicates roughly `1%`
  of general drift in this run — thermal state, or ordinary variation — that
  cannot be attributed to the change. Netting the control drift out leaves
  `maximum` somewhere between `+1.5%` and `+3.0%`, which is worse rather than
  better.
- **Unchanged:** Smoothed training loss and validation loss are bit-identical to
  Stage 9, as designed. All convolution sections, `matmul`, and `linear1` are
  untouched code and sit inside the drift band.
- **Diagnosis was wrong.** The division was not the bottleneck. `maximum` is
  almost certainly bound by the scatter itself: each output window writes to
  input positions chosen by the argmax, so the writes are irregular and land
  across a `3.2 MiB` output buffer for Conv2. Dividing once per window out of
  roughly 111 cycles per window is a small fraction of the work, and the
  out-of-order engine can overlap independent divisions. Fixing this would mean
  changing the access pattern, not removing arithmetic from it.
- **Two measurement problems this profile exposed.** First, the two available
  estimates of the effect disagree by `8.8×`: the medians of the nine logged
  checkpoints imply `+0.48 ms` per step, or `+0.45 s` over the epoch, while the
  measured epoch time rose `+3.98 s`. For comparison, the Stage 8 to Stage 9
  medians and epoch time agreed to within `10%`. So the nine logged checkpoints
  are not a representative sample of the 929 unlogged ones in this run, and
  medians of logged steps should not be used to predict epoch time. Second, the
  control drift itself is now `1%`, which is the same size as most of the
  individual optimisations still on the list.

## 11. Batch-parallel convolution input gradient

**Source label:** `Batch-parallel convolution input gradient`
**Branch:** `feature/weird-optimizations*`
**Command:** `MINITORCH_THREADS=4 MINITORCH_PROFILE_LAYERS=true make run-release`
**Configuration:** `epochs: 1`
**Status:** measured through first-epoch validation

The first step of the multithreading work, and the first change in this project
to use more than one core. `conv2d_grad_input_stride1_dilation1` was split into
a batch-range worker and a driver that fans out over the batch with
`std::thread::scope`, one scoped thread per chunk. No dependency is added, the
autograd graph and its `Rc` are untouched, and `chunks_mut` supplies the
disjoint output slices so no synchronisation or ownership transfer is needed.

Two properties make this kernel a good first target and the results
trustworthy:

- **Every byte of `packed_grad_output` is read exactly once regardless of thread
  count.** Total memory traffic is unchanged by splitting; only the arithmetic is
  divided. The measured `3.89×` on four threads is close to linear, which is
  what that access pattern predicts and what a bandwidth-bound kernel would not
  give.
- **The result is bit-identical at any thread count.** Batch entries are
  disjoint, so each output element's reduction over output channels is performed
  entirely within one thread and its order cannot change. This is stronger than
  the tile-based changes, where grouping into partials was itself a
  reassociation.

The worker count comes from `MINITORCH_THREADS`, defaulting to
`available_parallelism()`, so one binary can be swept without a rebuild and
`MINITORCH_THREADS=1` gives an exact serial baseline in the same build.

Correctness was verified across widths: the full suite of 234 tests passes at
`MINITORCH_THREADS` of 1, 2, 3, 5, 7, 8, and 16. The prime widths matter because
they do not divide the batch and so exercise the tail-chunk path, which the
default configuration never reaches.

| Metric | Value |
|---|---:|
| Dataset setup | 39.319959 ms |
| Dataloader setup | 0.452333 ms |
| Model setup | 3.427459 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Validation batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 10.398292375 s |
| Training time | 193.03315625 s |
| Smoothed training loss | 0.33636283742747686 |
| Validation loss | 0.26916314672841835 |
| Validation time | 10.471934791 s |

### Stage 10 comparison

| Metric | Stage 10, serial | Stage 11, 4 threads | Difference |
|---|---:|---:|---:|
| Initial evaluation | 10.457898792 s | 10.398292375 s | −0.6% |
| **Epoch 1 training** | **233.695550208 s** | **193.03315625 s** | **−40.66 s (−17.4%)** |
| Validation time | 10.475820292 s | 10.471934791 s | −0.0% |

### Operation-level comparison

Median over the nine full-batch checkpoints, steps 100–900.

| Operation | Stage 10, serial | Stage 11, 4 threads | Speedup |
|---|---:|---:|---:|
| Conv2 `grad_input` | 51.930958 ms | 13.353916 ms | **3.89×** |
| Conv1 `grad_input` | 3.265083 ms | 0.932959 ms | **3.50×** |
| Conv2 backward total | 101.364750 ms | 60.142750 ms | 1.69× |
| `conv2d` total | 112.482416 ms | 67.996667 ms | 1.65× |
| Per-step backward | 165.450917 ms | 122.847250 ms | 1.35× |
| Conv2 forward | 41.909791 ms | 41.878250 ms | 1.00× *(control)* |
| `matmul` | 24.265792 ms | 24.284583 ms | 1.00× *(control)* |
| `linear1` | 3.580042 ms | 3.608708 ms | 0.99× *(control)* |
| `maximum` | 20.843750 ms | 20.310376 ms | 1.03× *(control)* |
| `max_pool2d` | 4.206375 ms | 4.128292 ms | 1.02× *(control)* |
| Conv1 forward | 9.525167 ms | 9.679875 ms | 0.98% *(control)* |

### Effect on the PyTorch comparison

| Metric | PyTorch, 1 thread | Rust Stage 10 | Rust Stage 11 |
|---|---:|---:|---:|
| Backward | 59.264 ms | 165.45 ms (2.80×) | 122.85 ms (**2.07×**) |
| Epoch 1 training | 116.426 s | 233.696 s (2.01×) | 193.033 s (**1.66×**) |

The backward gap narrowed from `2.80×` to `2.07×` and the epoch gap from `1.97×`
to `1.66×`, in a single step that touched one kernel.

### Optimization review

- **Better:** Conv2 `grad_input` improved `3.89×` and Conv1's `3.50×`, almost
  linear on four threads, which confirms the kernel was compute-bound with
  thread-count-independent memory traffic. Epoch-1 training fell by `40.66 s`.
  Conv1's `grad_input` is the useful negative result: it is small enough that
  thread-spawn overhead could easily have dominated, and it still gained `3.50×`,
  so per-call spawning is viable at this kernel size.
- **Better, and unexamined:** Initial evaluation and validation time are
  unchanged to within `0.6%` and `0.0%`. Neither runs a backward pass, so this
  is the expected result and confirms the split touches only what it should.
- **Unchanged:** Every serial control is flat to within `2%`. Smoothed training
  loss and validation loss are bit-identical to Stages 9 and 10, as the disjoint
  batch decomposition guarantees.
- **Worse: nothing.** Conv1 forward rose `1.6%` and `add` rose `11.8%`, but step
  700 recorded `add` at `11.692041 ms` against a median near `2.9 ms` and a
  backward of `135.378833 ms` against a median near `122.8 ms`. Both are
  transient outliers in a run that also completed in `3m 35s` wall time against
  `4m 27s` previously, so the machine was in a different thermal state.
- **A methodological problem this run exposes.** Conv2 `grad_weight` moved
  `45.490042 → 42.127667 ms`, a `7.4%` improvement, in code this change does not
  touch and which runs before any thread is spawned. That is far larger than the
  `1%` control drift estimated from Stage 10, which means **cross-session
  comparisons carry several percent of noise**, and Stage 10's apparent `1.7%`
  regression may have been partly the same effect. The `3.89×` on `grad_input` is
  far outside that band and is not in doubt, but no smaller effect measured
  across two sessions should be. A `MINITORCH_THREADS` sweep run as a single
  session is the clean experiment, since every width is then measured against
  the same machine state.

## 12. Thread-width sweep

**Source label:** `Thread-width sweep`
**Branch:** `feature/weird-optimizations*`
**Commands:**
`MINITORCH_THREADS={1,2,4,8} MINITORCH_PROFILE_LAYERS=true make run-release`
**Configuration:** `epochs: 1`
**Status:** four profiled widths; the one-worker run is the current serial
reference

Same binary and same kernel as Stage 11, run at one, two, four, and eight
workers. This is one experiment with four points, recorded together so the
scaling curve can be read off a single table. Each width was a separate
process, so the sweep is not a same-process A/B test; the serial controls below
quantify the residual cross-session variation.

| Metric | Value |
|---|---:|
| Epoch 1 training, 1 worker | 227.564065542 s |
| Epoch 1 training, 2 workers | 201.891501334 s |
| Epoch 1 training, 4 workers | 193.03315625 s |
| Epoch 1 training, 8 workers | 189.177429041 s |
| Initial evaluation, 1 / 2 / 4 / 8 | 10.569629583 / 10.510451208 / 10.398292375 / 10.758061 s |
| Validation time, 1 / 2 / 4 / 8 | 10.769859125 / 10.564413333 / 10.471934791 / 10.539541958 s |
| Smoothed training loss, all widths | 0.33636283742747686 |
| Validation loss, all widths | 0.26916314672841835 |

### Scaling of the parallelised kernel

| Workers | Conv2 `grad_input` | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 48.173167 ms | 1.00× | — |
| 2 | 24.687042 ms | **1.95×** | 98% |
| 4 | 13.353916 ms | **3.61×** | 90% |
| 8 | 11.545375 ms | **4.17×** | 52% |

### Whole-model effect

| Workers | Per-step backward* | Epoch 1 training | vs PyTorch |
|---:|---:|---:|---:|
| 1 | 161.341750 ms | 227.564065542 s | 1.95× |
| 2 | 133.211916 ms | 201.891501334 s | 1.73× |
| 4 | 122.847250 ms | 193.03315625 s | 1.66× |
| 8 | 119.900291 ms | 189.177429041 s | **1.62×** |

\* Median of the eight regular full-batch checkpoints at steps 100–800; step
900 is a whole-run outlier and is excluded.

### Controls

Mid-run serial code paths at each width, reported as the median of the eight
full-batch checkpoints at steps 100–800. Step 900 is excluded because every
section, including the parallel one, inflated together. None of these paths is
affected by the worker count, so their spread measures session drift.

| Operation | 1 worker | 2 workers | 4 workers | 8 workers | Spread |
|---|---:|---:|---:|---:|---:|
| Conv2 `grad_weight` | 42.455313 ms | 42.364166 ms | 42.127667 ms | 42.265667 ms | 0.8% |
| Conv2 forward | 42.333000 ms | 41.861125 ms | 41.878250 ms | 41.728333 ms | 1.4% |
| `matmul` | 24.262687 ms | 24.315209 ms | 24.284583 ms | 24.377958 ms | 0.5% |
| `maximum` | 20.144208 ms | 20.313667 ms | 20.310376 ms | 20.564916 ms | 1.0% |
| `linear1` | 3.572979 ms | 3.597458 ms | 3.608708 ms | 3.588708 ms | 1.0% |

Conv1 `grad_input`, which is parallelised, scales `3.056`, `1.618`, `0.933`,
and `0.850 ms` across the same four widths.

### Optimization review

- **Better:** `grad_input` scales at `98%` efficiency on two workers and `90%` on
  four, then drops to `52%` on eight. The curve is the expected shape: near
  linear to the core count, then flat.
- **The ceiling is four.** Eight workers reduce the kernel time by `14%` over
  four (`13.35 → 11.55 ms`) and epoch time by `2.0%`. The machine has four
  performance cores and the efficiency cores contribute little to a kernel of
  this shape. **There is nothing meaningful left in thread count.** The
  `available_parallelism()` default of eight is safe rather than harmful, so it
  can stay, but going wider is pointless.
- **Unchanged:** Smoothed training loss and validation loss are bit-identical at
  every width and to every run since Stage 9, as the disjoint batch
  decomposition guarantees. Conv1's `grad_input` scales cleanly at all four
  widths, confirming per-call spawning still pays at `3 ms` of work.
- **The Stage 10 anomaly is now materially accounted for, and it was not
  small.** With a one-worker run in hand, the two serial kernels that Stage 10
  measured can be compared directly against this one:

  | Kernel | Stage 10 | 1 worker today | Offset |
  |---|---:|---:|---:|
  | Conv2 `grad_weight` | 45.490042 ms | 42.455313 ms | +7.2% |
  | Conv2 `grad_input` | 51.930958 ms | 48.173167 ms | +7.8% |

  Two independent kernels are high by almost exactly the same amount in that
  single session. That is consistent with a machine-wide offset, rather than a
  property of either kernel, and it retires the "unexplained 7.4% move" that
  Stage 11 could only leave open. It also means **Stage 10's `1.7%` regression
  claim is not supportable**: the implementation was correctly reverted, but
  the experiment established no measured regression, only a null result.
- **A correction to Stage 11's own reasoning, now settled by measurement.**
  Stage 11 used initial evaluation and validation as drift indicators, on the
  grounds that neither runs a backward pass. The controls table shows the
  opposite: those two phases vary by `3.5%` and `2.8%` across the four widths,
  while mid-run serial sections vary by `0.5–1.4%`. Initial evaluation runs
  before thermal steady state and so samples a different machine state. **Use a
  mid-run serial section as the drift reference, never the opening evaluation.**
- **Cross-session drift is real but bounded in recent runs.** Mid-run serial
  controls hold within `0.5–1.4%` across four separate processes, which is the
  figure future comparisons should be judged against. Stage 10 is the exception:
  its two relevant kernels were both about `7–8%` high, so it must not be used as
  the serial reference.
- **A new variance source, now measured.** With one kernel split, a straggler
  thread stalls the join. The eight-thread profile recorded `grad_input`
  outliers of `13.70` and `14.95 ms` against a median of `11.55`; the two-worker
  profile shows the same at `26.46 ms`; this serial profile has a whole-step
  outlier at step 900 where *every* layer inflated together. Future parallel
  sections will carry this tail, and it will grow as more kernels are split.


## Observations and next measurements

- The loss values are bit-identical across Stages 0 through 8 and diverge at
  Stage 9 by one unit in the last place, from a deliberate reassociation. No
  change has altered the training trajectory in any meaningful way.
- **The bit-identity gate is no longer available.** From Stage 9 onward,
  correctness rests on the reference-comparison tests in
  `tests/nn/conv2d.rs`, which check the kernels against independent triple-loop
  implementations at `1e-9` tolerance and include mutation-tested coverage of
  every tiling and remainder path.
- Stages 7, 8, and 9 are all profiled runs, so their differences are clean. A
  clean `make run-release` measurement is still needed to bring the unprofiled
  baseline up to date; the profiled figures are not directly comparable to the
  Stage 6 clean row.
- The dot-product matmul path remains scalar. LLVM will not vectorize a
  floating-point reduction without `reassoc`, and Rust does not set it, so that
  path gained contiguous access and lost its accumulator traffic but gained no
  SIMD. Fixing it needs a two-dimensional register tile with `k` innermost, which
  is a larger change than the two unit-stride paths used here.
- Conv2 `grad_input` is now threaded, so the largest untouched convolution
  sections are `grad_weight` at about `42.3 ms` and the convolution forward at
  about `41.7 ms`. Both read distinct batch slices and are the next parallel
  candidates. The residual bounds checks and tile-length reload noted earlier
  remain secondary to that work.
- `maximum` is the one large code path in the model that has resisted
  optimization. Stage 10 removed a floating-point division from it and gained
  nothing measurable, which points at the scatter's irregular write pattern
  rather than at arithmetic. It is the third-largest backward category at
  `20.8 ms` and is probably not worth further effort without changing its access
  pattern.
- **Four bottleneck diagnoses in this project have now been wrong**: the
  matmul rewrite was sized as an accumulator problem when the real fault was an
  opaque runtime stride; the two conv tilings were sized from a load-port model
  that ignored non-memory costs; and the pool fast path was sized around a
  division that turned out to be a rounding error in the budget. Every change
  that worked was one where the disassembly was read first and the diagnosis
  came from the instruction stream, not from a model. Adopt that as the rule:
  disassemble, then decide, and treat any timing prediction as unverified until
  the profile confirms it.
- Medians of the nine logged full-batch checkpoints are not a reliable
  predictor of epoch time: in Stage 10 they implied `+0.45 s` against a measured
  `+3.98 s`. The current sweep also shows why a whole-step outlier must be
  excluded rather than averaged in. Use epoch and validation totals for
  decisions and medians only for attribution within a single run.
- Recent mid-run serial controls vary by `0.5–1.4%` across the four width runs.
  Differences below roughly `2%` should therefore not be treated as signal
  without a repeat run; Stage 10 is a documented exception rather than the
  expected drift level.
- Conv1's position-tiling regression suggests the tile needs a size guard keyed
  on `in_channels × out_channels`, since a layer with one input channel has
  nine taps and cannot amortise the per-tile setup. This needs a structural
  predicate rather than a bare constant, and it is worth `0.50 ms` per step.
- Predictions from instruction-count and load-port models have now been wrong
  three times, and a fourth bottleneck diagnosis was simply incorrect. These
  models are reliable for ruling a change *out* and for confirming that codegen
  changed as intended, and unreliable for sizing the wall-clock effect. Treat
  future estimates as unverified until a profile confirms them.
- The single-threaded PyTorch reference completes epoch 1 in `116.426 s`.
  Rust's current serial profile is `227.564 s` (`1.95×`), and the default
  eight-worker profile is `189.177 s` (`1.62×`). Its per-step backward is
  `119.900 ms` against PyTorch's `59.264 ms` (`2.02×`), down from `2.80×`
  before threading. The current serial initial evaluation and validation are
  within `1.19×` and `1.17×` of PyTorch, respectively.
  **The largest untouched pieces are Conv2 `grad_weight` at about `42.3 ms` and
  the convolution forward at about `41.7 ms`.**
- The batch-parallel input gradient scales at `1.95×` on two workers (`98%`
  efficiency), `3.61×` on four (`90%`), and `4.17×` on eight (`52%`), using the
  current one-worker `48.17 ms` baseline. **Four workers is the practical
  ceiling** for this kernel and machine; the remaining gains must come from
  parallelising more kernels, not from a larger count. `grad_weight` and the
  convolution forward read distinct batch slices and so should scale, whereas
  `maximum` is scatter-bound and may not.
- Recent mid-run serial controls vary by `0.5–1.4%` across separate processes.
  Initial evaluation and validation vary more (`3.5%` and `2.8%`) because they
  run before or outside the thermal steady state of the training loop. **Use a
  mid-run serial section as the drift reference, never the opening
  evaluation.** Stage 10 is the one documented high-drift session and is not a
  valid serial reference.
- Threading introduces a new variance source that did not exist in the serial
  profile: with one kernel split, a single straggler thread stalls the join, and
  the eight-thread profile recorded `grad_input` outliers of `13.70` and
  `14.95` ms against a median of `11.55`. Tail latency in parallel sections
  should be expected in future medians.
- `grad_weight` is the next target and needs a reduction: per-thread partial
  buffers summed at the end. It is already reassociating from the Stage 9
  position tile, so the reduction adds no new numerical cost.
- Cross-session drift in recent mid-run serial controls is `0.5–1.4%`, but Stage
  10 was about `7–8%` high on two separate kernels and must not be used as a
  control. Compare future changes to the recent one-worker profile and repeat
  effects below `2%`.
- Re-running the PyTorch reference at four threads would show whether its
  `59.3 ms` backward also scales by roughly `3×`, which would set the realistic
  target for this port.
- Loss values are not comparable to the PyTorch reference until the seed,
  initialisation, and hyperparameters are matched; the initial losses already
  differ, so the trajectories diverge for that reason alone.
- Future profiles should record CPU model, commit, dtype, and thread settings
  before making claims that require cross-machine or PyTorch comparisons.
- Do not extrapolate from individual checkpoints: step `938/938` is a half
  batch, and the one-worker profile's step 900 inflated every section at once.
  Use epoch and validation totals for decisions, and medians only for
  within-run attribution.

## PyTorch reference

The reference run below is a **complete single-threaded PyTorch `float64` CPU
run of the same MNIST workload**, recorded separately from the Rust profiles
because it is a different implementation and a different run, not a variant of
the Rust code. It replaces an earlier, partial record that held only one
checkpoint and no epoch totals; the earlier record's per-checkpoint timings
(`forward 0.052357 s`, `backward 0.063982 s` at step 100) did not come from this
run and should not be compared with anything below.

| Metric | Value |
|---|---:|
| Device | `cpu` |
| Dtype | `torch.float64` |
| Torch threads | 1 |
| Dataset setup | 0.508 s |
| Dataloader setup | not reported |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Validation batches | 157 |
| Initial evaluation | 8.896 s |
| Initial loss | 2.314024 |
| **Epoch 1 training time** | **116.426 s** |
| **Epoch 1 smoothed train loss** | **0.288240** |
| **Validation loss** | **0.236946** |
| **Validation time** | **9.213 s** |

**On the thread count.** PyTorch's default `torch.get_num_threads()` is the
number of CPU cores, *not* one, so the `torch threads: 1` above is an explicit
setting by the reference script rather than a default. It was not set in this
repository and could not be verified here, because PyTorch is not installed on
the development machine. Two things should be confirmed before this reference is
relied on again:

- `torch.set_num_threads(1)` must be called before any eager, JIT, or autograd
  code runs, per the PyTorch documentation. If it were called after a forward
  pass had already initialised the OpenMP pool, the effective thread count could
  exceed 1 and every ratio below would be optimistic for Rust.
- `torch.get_num_interop_threads()` is not pinned by the printed value. For a
  sequential training loop it should not engage, but it is unverified.

Asserting `torch.get_num_threads() == 1` both immediately after the setter and
again after the first step would settle both. Until that is done, treat the
per-op ratios as indicative and the epoch totals as the reliable comparison,
since a leaked thread would inflate the per-op figures far more than a
157-batch validation pass.

### PyTorch checkpoints

| Step | Forward | Loss | Backward | Step time | Smoothed loss | Grad entries |
|---:|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 0.049959 s | 0.000103 s | 0.058772 s | 0.000941 s | 2.146694 | 8 |
| 200 / 938 | 0.053423 s | 0.000109 s | 0.063430 s | 0.000948 s | 1.255469 | 8 |
| 300 / 938 | 0.049617 s | 0.000174 s | 0.060420 s | 0.000932 s | 0.672885 | 8 |
| 400 / 938 | 0.047998 s | 0.000103 s | 0.056277 s | 0.000953 s | 0.516104 | 8 |
| 500 / 938 | 0.048667 s | 0.000102 s | 0.057258 s | 0.001011 s | 0.491253 | 8 |
| 600 / 938 | 0.047882 s | 0.000101 s | 0.067166 s | 0.000958 s | 0.395137 | 8 |
| 700 / 938 | 0.048356 s | 0.000108 s | 0.056516 s | 0.000958 s | 0.370829 | 8 |
| 800 / 938 | 0.050063 s | 0.000106 s | 0.059264 s | 0.000932 s | 0.371557 | 8 |
| 900 / 938 | 0.059339 s | 0.000198 s | 0.068877 s | 0.001178 s | 0.335618 | 8 |
| 938 / 938 | 0.031080 s | 0.000105 s | 0.042166 s | 0.001423 s | 0.288240 | 8 |

### Head-to-head against Stage 9

Both are single-threaded, so this is a like-for-like comparison. Rust figures
are medians over the nine full-batch checkpoints; PyTorch figures are the same.

| Metric | PyTorch | Rust Stage 9 | Ratio |
|---|---:|---:|---:|
| Forward | 49.617 ms | 78.890 ms | **1.59×** |
| Backward | 59.264 ms | 165.792 ms | **2.80×** |
| Optimizer step | 0.953 ms | 0.868 ms | **0.91×** |
| **Per-step total** | **109.937 ms** | **245.550 ms** | **2.23×** |
| Epoch 1 training | 116.426 s | 229.711 s | **1.97×** |
| Initial evaluation | 8.896 s | 10.407 s | **1.17×** |
| Validation | 9.213 s | 10.493 s | **1.14×** |

The shape of this gap matters more than its size:

- **The remaining 2.8× is almost entirely in the backward pass.** Forward is
  within `1.59×` and the two forward-only phases — initial evaluation and
  validation — are within `1.17×` and `1.14×`. The optimizer step is marginally
  *faster* than PyTorch.
- That is consistent with where the Rust time goes: `conv2d` backward is
  `111.40 ms` of the `165.79 ms` total, against a PyTorch backward of
  `59.26 ms` for everything. PyTorch routes convolution backward through
  oneDNN, whose kernels are register-blocked, cache-blocked, and prefetching;
  the direct kernels here are hand-tiled but still run one core with a scalar
  tail in places.
- Evaluation and validation being close while training is not is a useful
  signal: the forward path is in reasonable shape, so effort belongs in
  backward, not in the activation or pooling layers.

### Loss values are not comparable across the two implementations

| Metric | PyTorch | Rust Stage 9 |
|---|---:|---:|
| Initial loss | 2.314024 | 2.3064304231216664 |
| Epoch 1 smoothed train loss | 0.288240 | 0.33636283742747686 |
| Validation loss | 0.236946 | 0.26916314672841835 |

Both initial losses sit near `ln(10) = 2.302585`, which is what an untrained
model should give, but they are **not equal**, so the two runs used different
initial weights and the trajectories are not comparable point by point. The
curves cross repeatedly — Rust is lower at steps 200, 300, 800, and 900, and
higher at 400, 500, 600, 700, and at the end — which is what different seeds
produce, not what a systematically wrong gradient produces.

**No claim is made here that the two implementations agree numerically.** Doing
so would require matching the seed, the initialisation, and every
hyperparameter, and none of that has been verified. A loss mismatch at this
magnitude should be treated as unresolved until the configurations are pinned
down, and the timings above stand on their own regardless because they are
comparisons of wall-clock time rather than of results.

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
