# MNIST Optimization History

This file records benchmark results for the Rust `minitorch` implementation and
its PyTorch reference. Results are kept even when a run is incomplete so that
regressions and interrupted measurements remain visible.

## Benchmark protocol

Use the same configuration for comparable runs:

```text
config/train_config.yml
batch_size: 64
epochs: 5
seed: 42
log_every: 100
```

Recommended commands:

```bash
cargo run --release --bin mnist
python main/mnist_pytorch.py --device cpu --dtype float64 --num-threads 1
```

For CPU comparisons, record:

- Rust and PyTorch versions/commits
- CPU model and operating system
- Build profile (`release`)
- PyTorch thread count and dtype
- Dataset setup time
- Initial evaluation time
- Per-batch forward, loss, backward, and optimizer times
- Complete training epoch time
- Validation time
- Number of batches and samples

A run interrupted before completing an epoch is marked as incomplete; its
partial numbers are not treated as full-epoch results.

## Reference: PyTorch CPU

**Status:** measured reference run

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

The sum of the displayed per-operation times is approximately `0.117414 s`
per batch. Multiplying by 100 gives an extrapolated `11.7414 s` for 100
batches; this is derived from the displayed timings, not a separately measured
100-batch wall-clock interval.

## Stage 0: Rust baseline

**Status:** measured, incomplete run

This is the first available Rust MNIST run, before the `AutogradMeta` enum and
before the direct `Conv2d` kernels. The run was interrupted at step 300.

| Step | Forward | Loss | Backward | Optimizer | Grad entries |
|---:|---:|---:|---:|---:|---:|
| 100 / 938 | 382.490125 ms | 0.016959 ms | 531.922417 ms | 0.954208 ms | 8 |
| 200 / 938 | 377.973750 ms | 0.019792 ms | 538.015541 ms | 0.823042 ms | 8 |
| 300 / 938 | 389.378041 ms | 0.016584 ms | 594.515042 ms | 1.045625 ms | 8 |

Approximate displayed per-batch totals:

| Step | Forward + loss + backward + optimizer |
|---:|---:|
| 100 | 915.384 ms |
| 200 | 916.832 ms |
| 300 | 984.955 ms |

Not recorded for this interrupted run:

- Complete epoch time
- Validation time
- Final loss
- Full five-epoch runtime

## Stage 1: `AutogradMeta` enum and explicit `no_grad`

**Status:** measured, partial run

Reported command:

```bash
make run-release
```

Build and configuration:

| Metric | Value |
|---|---:|
| Build | Rust release profile |
| Dataset setup | 27.238250 ms |
| Dataloader setup | 0.493084 ms |
| Model setup | 3.963084 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 58.242783708 s |

This run establishes the post-`AutogradMeta` initial-evaluation measurement.
No training-step or complete-epoch measurements from this run were supplied.

The initial evaluation is approximately 6.24× slower than the PyTorch
reference (`58.2428 s / 9.336 s`), although this is only a cross-implementation
comparison: the models use different initialization streams and the Rust
implementation is scalar/single-threaded while PyTorch uses optimized native
kernels.

## Stage 2: Fused `Conv2d` kernels

**Status:** measured, partial run

Reported command:

```bash
make run-release
```

The run was supplied under the label “after the Fused `Conv2d` kernels.” The
history previously described the direct-kernel implementation as keeping bias
separate, so the exact bias-fusion boundary should be confirmed against the
commit used for this run.

| Metric | Value |
|---|---:|
| Build | Rust release profile |
| Dataset setup | 25.880375 ms |
| Dataloader setup | 0.366584 ms |
| Model setup | 2.896334 ms |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Training batches | 938 |
| Test batches | 157 |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 94.987133625 s |

Compared with the post-`AutogradMeta` initial evaluation:

```text
94.987133625 s - 58.242783708 s = +36.744349917 s
94.987133625 / 58.242783708 ≈ 1.631×
```

This run is therefore an initial-evaluation regression of approximately
63.1% relative to the post-`AutogradMeta` run. No training-step, complete-epoch,
or validation measurements from this run were supplied.

## Stage 2b: Packed-weight and output-channel-reuse forward kernel

**Status:** measured, partial run

This run adds the forward-loop optimization after the initial direct-kernel
regression: weights are packed for contiguous output-channel access, padding
is materialized once, and each input value is reused across output channels.
Bias remains separate, and the direct backward kernel is unchanged.

| Metric | Value |
|---|---:|
| Build | Rust release profile |
| Dataset setup | 25.406459 ms |
| Dataloader setup | 0.347958 ms |
| Model setup | 4.042083 ms |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 19.9548625 s |

### Step 100

| Operation | Time |
|---|---:|
| Forward | 123.757750 ms |
| Loss | 0.018667 ms |
| Backward | 509.155250 ms |
| Optimizer step | 0.581709 ms |
| Smoothed loss | 2.0968491171052857 |

The displayed per-batch total is approximately `633.513 ms`. Compared with the
previous direct-kernel run, initial evaluation improved from `94.9871 s` to
`19.9549 s` (about `4.76×` faster). Compared with the post-`AutogradMeta` run,
it is about `2.92×` faster. The remaining dominant cost is backward
(`509.155 ms` at step 100), which has not yet received the same loop/packing
optimization.
## Stage 2c: Packed-weight and output-channel-reuse backward kernel

**Status:** measured, partial run

This run applies the same packed-weight/output-channel-reuse strategy to the
backward kernel. The run was stopped after step 100, so no complete-epoch or
validation measurement is included.

| Metric | Value |
|---|---:|
| Build | Rust release profile |
| Dataset setup | 30.106625 ms |
| Dataloader setup | 0.379584 ms |
| Model setup | 3.688834 ms |
| Initial loss | 2.3064304231216664 |
| Initial evaluation | 20.150486583 s |

### Step 100

| Operation | Time |
|---|---:|
| Forward | 125.472667 ms |
| Loss | 0.016709 ms |
| Backward | 214.608250 ms |
| Optimizer step | 0.835417 ms |
| Smoothed loss | 2.0968491171052857 |

The displayed per-batch total is approximately `340.933 ms`. Compared with
Stage 2b, backward decreased from `509.155250 ms` to `214.608250 ms`, a
reduction of approximately `57.9%`. Initial evaluation is within normal
run-to-run variation of the optimized-forward result (`19.9548625 s`).

## Stage 3: Additional bias-fusion measurements

**Status:** not separately benchmarked

If the Stage 2 run used the direct kernel with bias still separate, the next
optimization is to initialize or accumulate bias in the convolution kernel and
compute its gradient in the convolution backward rule. Keep that measurement
separate so the source of any timing change is clear.

## Result-entry template

When adding a run, preserve the original output and fill this structure:

```markdown
### <implementation> — <YYYY-MM-DD>

- Commit:
- Device:
- Dtype:
- Threads:
- Build:
- Dataset setup:
- Initial evaluation:
- Training batches:
- Validation batches:

| Step | Forward | Loss | Backward | Optimizer | Smoothed loss |
|---:|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... |

- Training epoch time:
- Validation time:
- Final/validation loss:
- Notes:
```

## Data availability note

The available measurements are the PyTorch reference, the interrupted Rust
baseline, the partial post-`AutogradMeta` run, the initial direct/fused
`Conv2d` run, the packed-weight/output-channel-reuse forward run, and the
packed-weight/output-channel-reuse backward run. No complete training-epoch
measurement or separately identified bias-fusion benchmark has been supplied
yet, so those measurements are intentionally left pending.
