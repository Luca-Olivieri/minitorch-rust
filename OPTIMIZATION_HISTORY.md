# MNIST Optimization History

This file records the Rust MNIST `SmallCNN` optimization measurements. The
four Rust profiles below are transcribed from the recorded optimization runs;
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
- `grads.len() = 8` was reported at every checkpoint in all four profiles.
- Record CPU model, operating system, exact commit, dtype, and thread settings
  with each profile when making a cross-machine comparison.

## Summary

All four runs use the same dataset sizes reported by the program: 60,000
training samples, 10,000 test samples, 938 training batches per epoch, and 157
validation batches.

| Profile | Branch/label | Initial evaluation | Epoch 1 training | Epoch 1 smoothed loss | Validation loss | Validation time |
|---|---|---:|---:|---:|---:|---:|
| Baseline | `main` / `0. Baseline` | 56.488265041 s | 868.228344666 s | 0.3363628374274766 | 0.2691631467284183 | 57.073553542 s |
| After `AutogradMeta` | `feature/weird-optimizations~2` | 60.536313709 s | 889.215288917 s | 0.3363628374274766 | 0.2691631467284183 | 58.738805208 s |
| After fused forward/backward `Conv2d` kernels | `2. after fused fw and bw Conv2d kernels` | 20.130340208 s | 328.818958666 s | 0.3363628374274767 | 0.2691631467284183 | 20.493247959 s |
| After compact max-pool metadata | `feature/weird-optimizations*` | 15.5961365 s | 308.258938459 s | 0.3363628374274767 | 0.2691631467284183 | 15.865784084 s |

The four profiles follow the same measurement boundary: initial evaluation,
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

## Observations and next measurements

- The first-epoch training loss and validation loss remain effectively identical
  across all four profiles, so the timing changes did not alter the observed
  training trajectory.
- The compact max-pool metadata profile is the current performance reference.
  Relative to the direct convolution-kernel profile, it reduced complete
  first-epoch training time from `328.818958666 s` to `308.258938459 s`,
  initial evaluation from `20.130340208 s` to `15.5961365 s`, and validation
  time from `20.493247959 s` to `15.865784084 s`.
- Future profiles should record CPU model, commit, dtype, and thread settings
  before making claims that require cross-machine or PyTorch comparisons.
- Do not extrapolate from individual checkpoints, especially step `938/938`;
  use the measured epoch and validation totals for comparisons.

## Historical PyTorch reference

The following reference is kept separately from the four Rust profiles above.

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
