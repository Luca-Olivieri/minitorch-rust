# Conv2d Computation: Current State and Optimization Options

This document describes the current direct `Conv2d` implementation in terms of
**data access, allocations, and cache locality**, then evaluates potential
optimizations using the same criteria.

The scope is the groups-one, first-order `Conv2d` used by the MNIST `SmallCNN`.
The implementation lives primarily in
[`src/core/storage/ops/conv.rs`](src/core/storage/ops/conv.rs) and is invoked by
[`src/core/autograd/ops/conv.rs`](src/core/autograd/ops/conv.rs).

## 1. Workload and notation

The model uses two convolutions:

| Layer | Input shape (`f64`, batch 64) | Weight shape | Output shape |
|---|---|---|---|
| Conv1 | `[64, 1, 28, 28]` | `[1, 32, 3, 3]` | `[64, 32, 28, 28]` |
| Conv2 | `[64, 32, 14, 14]` | `[32, 64, 3, 3]` | `[64, 64, 14, 14]` |

Both layers use:

```text
kernel   = 3 × 3
stride   = 1
dilation = 1
padding  = Same
dtype    = f64
groups   = 1
```

Notation used below:

- `B`  — batch size
- `CI` — input channels
- `CO` — output channels
- `H`, `W` — input spatial dimensions
- `OH`, `OW` — output spatial dimensions
- `KH`, `KW` — kernel dimensions
- `OC` — one output-channel vector, length `CO`

The latest clean unprofiled baseline completed epoch 1 in approximately
`299.386 s`. The most recent detailed profile before the reverted square-
kernel experiment attributed roughly `65–67 ms` to Conv2 `grad_input`,
`51–54 ms` to Conv2 `grad_weight`, and `42–45 ms` to matmul backward. Those
section timings are diagnostic; the current code should be re-profiled after
any change.

## 2. Current storage and ownership model

`TensorStorage<T>` owns an `Rc<Vec<T>>`, shape, strides, and an offset. Views
can share the underlying buffer, while `from_buffer` takes ownership of a fresh
`Vec<T>`.

Consequences for convolution:

- Input and weight storages are shared with the autograd graph through handles.
- The convolution result receives a fresh output buffer.
- Temporary packed and padded buffers are local to the storage kernel and are
  dropped when the operation returns.
- The autograd rule stores convolution parameters and the backward marker, not a
  cached packed-weight copy between calls.
- The optional profiler adds a small timing collector, but the normal path does
  not retain profiling buffers after the operation.

## 3. Current forward convolution

### 3.1 Data transformations

For each forward call, `conv2d` currently performs:

1. Shape and parameter validation.
2. Padding materialization when any padding is nonzero.
3. Weight packing from `[CI, CO, KH, KW]` to `[CI, KH, KW, CO]`.
4. Output allocation and accumulation.

For Conv2 with `Same` padding:

```text
original input: 64 × 32 × 14 × 14  = 401,408 f64 values ≈ 3.06 MiB
padded input:   64 × 32 × 16 × 16  = 524,288 f64 values = 4.00 MiB
packed weight:  32 × 3 × 3 × 64      = 18,432 f64 values ≈ 144 KiB
output:         64 × 64 × 14 × 14  = 802,816 f64 values ≈ 6.13 MiB
accumulator:    64 f64 values = 512 bytes
```

Conv1 has a much smaller input and weight working set, but its output is about
`12.25 MiB` because it has 32 channels at `28 × 28` resolution.

### 3.2 Loop order and data access

The forward loop is:

```text
for batch
  for output row
    for output column
      zero the CO-element accumulator
      for input channel
        for kernel row
          for kernel column
            read one input value
            for output channel
              accumulator[oc] += input_value * packed_weight[tap, oc]
      write the accumulator to the NCHW output
```

Data-access properties:

- The input value is loaded once per `(b, ic, kh, kw, oh, ow)` and reused
  across all output channels.
- `packed_weight[tap, 0..CO]` is contiguous.
- `accumulator[0..CO]` is contiguous and remains hot in a small working set.
- The output is written with a plane-strided access: consecutive `oc` values
  are separated by `OH × OW` elements in the NCHW buffer.
- The padded input is read through its strides; the MNIST storage is contiguous,
  but the kernel still uses the general strided indexing path.
- Weight packing is repeated on every forward call.

### 3.3 Allocations

Forward allocates or wraps:

- One padded input buffer when padding is nonzero.
- One packed weight buffer.
- One output buffer.
- One small `CO`-element accumulator reused for every spatial position.

The output and padded input dominate the memory traffic. The packed weight is
small enough to be cache-friendly, but it is still rebuilt for every call.

### 3.4 Cache locality

Positive properties:

- The inner output-channel loop is contiguous.
- The accumulator is small and reused.
- Each input value is reused across all output channels.
- Packed weights avoid repeated strided reads of the original weight layout.

Costs:

- Packing weights adds a read/write pass over the parameter buffer.
- Padding copies the input before the arithmetic pass.
- The output write is strided across channel planes.
- The direct loop has no explicit spatial or channel tiling.
- The inner loop relies on compiler autovectorization; there is no explicit
  SIMD blocking.

## 4. Current backward convolution

The backward rule calls one storage function that computes both the input and
weight gradients.

### 4.1 Data transformations and allocations

`conv2d_backward_impl` currently performs:

1. Pack the upstream output gradient from NCHW into pixel-major layout
   `[B, OH, OW, CO]`.
2. Pack the weight into `[CI, KH, KW, CO]`.
3. Materialize a padded input buffer.
4. Accumulate a packed weight gradient in `[CI, KH, KW, CO]` layout.
5. Accumulate the input gradient in logical NCHW order.
6. Unpack the weight gradient back to the parameter layout.

For Conv2, the principal buffers are approximately:

| Buffer | Shape | Size |
|---|---|---:|
| Packed upstream gradient | `[64, 14, 14, 64]` | `6.13 MiB` |
| Packed weight | `[32, 3, 3, 64]` | `144 KiB` |
| Padded input | `[64, 32, 16, 16]` | `4.00 MiB` |
| Packed weight gradient | `[32, 3, 3, 64]` | `144 KiB` |
| Input gradient | `[64, 32, 14, 14]` | `3.06 MiB` |
| Unpacked weight gradient | `[32, 64, 3, 3]` | `144 KiB` |

The packed upstream gradient and padded input are large enough to matter for
memory traffic. The weight buffers are small enough to remain cache-resident on
most CPUs.

### 4.2 Weight-gradient loop

The weight-gradient loop is:

```text
for batch
  for output row
    for output column
      for input channel
        for kernel row
          for kernel column
            read one padded input value
            for output channel
              packed_grad_weight[tap, oc] += input_value * packed_grad_output[pixel, oc]
```

Data-access properties:

- The input value is reused across `CO` output channels.
- The upstream-gradient vector is contiguous.
- The packed weight-gradient vector is contiguous.
- The same packed weight-gradient vector is updated repeatedly across batch and
  spatial positions.
- The packed weight buffer is small, so these read-modify-write updates have
  good locality.

Costs:

- The upstream-gradient vector is revisited for every input channel and tap.
- The input and upstream buffers are traversed separately from the input-
  gradient pass.
- The original parameter layout is restored in a final unpack pass.

### 4.3 Input-gradient loop

The input-gradient loop is input-stationary:

```text
for batch
  for input channel
    for input row
      for input column
        zero the CO-element accumulator
        for each valid kernel tap
          for output channel
            accumulator[oc] += packed_grad_output[pixel, oc] * packed_weight[tap, oc]
        write one input-gradient value
```

For unit stride and unit dilation, the current implementation uses direct
coordinate mapping:

- Interior positions avoid per-tap coordinate bounds checks.
- Boundary positions use a checked path.
- The input-gradient accumulator is one reusable `CO`-element buffer.
- The upstream-gradient and weight vectors are contiguous within each tap.

This is the section that currently dominates Conv2 backward, typically around
`65–67 ms` in the recent profile.

### 4.4 Backward cache locality

Positive properties:

- The `CO`-element accumulator stays hot.
- Upstream gradients and packed weights are contiguous along `CO`.
- The weight gradient buffer is small.
- Direct contiguous accumulation gives the compiler a straightforward
  autovectorization opportunity.

Costs:

- The packed upstream gradient is built before both gradient loops.
- The padded input is built before the weight-gradient loop.
- The weight-gradient and input-gradient passes traverse large buffers
  separately.
- Each input-gradient position gathers upstream gradients from several output
  positions.
- Neighboring input positions reuse the same weight vector, but the current
  loop does not explicitly tile those positions to exploit that reuse.
- There is no cache blocking across input channels, spatial positions, or
  output-channel groups.

## 5. Evaluated optimization experiments

### 5.1 Per-tap output-channel chunk blocking — reverted

This used `chunks_mut`/nested iterators to process small output-channel blocks
inside every kernel tap.

- **Data access:** Contiguous slices were preserved, but iterator construction
  occurred once per tap.
- **Allocations:** No large new persistent buffers, but substantial short-lived
  iterator/setup overhead.
- **Cache locality:** Intended to improve reuse, but the per-tap abstraction
  prevented the compiler from optimizing the hot loop as effectively.
- **Pros:** Conceptually simple output-channel blocking.
- **Cons:** Measured a large backward regression. It should not be reintroduced
  in this form.

### 5.2 Square-kernel specialization — reverted

This added a separate path for square kernels with unit stride and dilation.

- **Data access:** Used incremental packed-weight offsets and direct slices.
- **Allocations:** No meaningful allocation change.
- **Cache locality:** No clear improvement over the existing unit-stride path.
- **Pros:** Supports arbitrary square kernel sizes without a 3×3-specific
  implementation.
- **Cons:** The measured `grad_input` section did not improve; Conv2 backward
  and aggregate training were slightly worse. The path was removed.

## 6. Potential optimizations

Each option below uses the same criteria: data access, allocations, cache
locality, expected pros, and expected cons.

### 6.1 Reuse a parameter-resident packed weight

**Idea:** Pack each convolution weight once and reuse the packed representation
in forward and backward.

- **Data access:** Removes the per-call weight-packing pass. The forward and
  backward kernels read the same `[CI, KH, KW, CO]` layout directly.
- **Allocations:** Removes the packed-weight `Vec` from every forward and
  backward call. The packed representation would live with the module or a
  parameter wrapper.
- **Cache locality:** The packed weight is small (`144 KiB` for Conv2) and
  would remain hot across calls. This is unlikely to be the main bottleneck.
- **Pros:** Simple, low algorithmic risk; removes repeated reads/writes and
  makes weight layout explicit.
- **Cons:** The optimizer currently updates the original parameter layout, so
  packed weights must be kept synchronized or become the canonical parameter
  representation. This touches parameter ownership, optimizer paths, and
  serialization. Expected gain is small because the measured pack-weight
  section is already very small.

### 6.2 Avoid materializing padded input

**Idea:** Read the original input with interior/boundary-aware indexing instead
of constructing a padded copy.

- **Data access:** Interior taps read the original contiguous input. Boundary
  taps either use a zero check or a small boundary path.
- **Allocations:** Removes the Conv2 padded input buffer, approximately `4 MiB`
  per forward or backward call.
- **Cache locality:** Avoids writing and rereading a large temporary buffer,
  reducing cache pollution and one full memory pass.
- **Pros:** Lowers allocation and memory traffic; removes a distinct `padded
  input` profiling section.
- **Cons:** Adds bounds logic to the hot loops, complicates strided inputs, and
  can hurt vectorization if the compiler cannot hoist the boundary checks. The
  padded-input section is only around `2 ms` in the recent profile, so the
  expected arithmetic gain is modest.

### 6.3 Spatial/register tiling for `grad_input`

**Idea:** Process two or more neighboring input positions together while reusing
packed weights and upstream-gradient slices.

- **Data access:** A weight vector for one `(ic, kh, kw)` tap can be loaded
  once and applied to multiple input positions. Upstream-gradient slices remain
  contiguous per pixel.
- **Allocations:** Use a small fixed-size tile buffer allocated once per
  convolution call, or stack storage for a compile-time-sized tile. No large
  im2col matrix is required.
- **Cache locality:** Improves temporal reuse of packed weights and may keep a
  tile of accumulators hot. This is the most promising direct-kernel option
  because `grad_input` is the largest backward section.
- **Pros:** Directly targets the measured bottleneck; preserves the packed
  layout and generic stride/dilation behavior.
- **Cons:** Increases register pressure and code complexity. Boundary handling
  becomes tile-aware. A naive per-tap chunk implementation regressed badly, so
  the tile must use fixed buffers and direct slices rather than nested dynamic
  iterators.

### 6.4 Spatial tiling for `grad_weight`

**Idea:** Process neighboring output positions in a tile and reuse the packed
upstream-gradient and input values.

- **Data access:** Multiple output positions can share input-channel/tap loop
  setup and improve reuse of the small packed weight-gradient buffer.
- **Allocations:** Requires a small tile buffer or multiple accumulators, but no
  large im2col matrix.
- **Cache locality:** The packed weight gradient is already small; the main
  benefit would be reducing repeated traversal and improving temporal locality
  of the input/upstream buffers.
- **Pros:** Could reduce loop overhead and improve reuse without changing the
  gradient algorithm.
- **Cons:** The packed weight-gradient buffer is already cache-friendly, so
  the benefit may be smaller than for `grad_input`. Tiling can also increase
  write traffic if accumulators are not kept in registers.

### 6.5 Fuse `grad_input` and `grad_weight` computation

**Idea:** Traverse the convolution data once and update both gradient outputs.

- **Data access:** Read each input/upstream value once and reuse it for both
  gradient computations.
- **Allocations:** Could share packed inputs and avoid one of the two large
  temporary traversals, but likely needs additional tile buffers.
- **Cache locality:** Better overall reuse, but the two gradients have different
  update patterns: `grad_weight` accumulates by tap, while `grad_input`
  accumulates by input position.
- **Pros:** Potential reduction in memory traffic and loop setup.
- **Cons:** More complex control flow, possible register spills, and greater
  risk of harming the currently vectorizable `grad_input` loop. This is a later
  experiment, not the first tiling attempt.

### 6.6 Im2col plus a custom blocked GEMM

**Idea:** Materialize all convolution patches as matrix columns, reshape weights
to `[CO, CI × KH × KW]`, and call a custom cache-blocked GEMM.

- **Data access:** Converts irregular patch gathers into large contiguous matrix
  operations. The GEMM can use register tiles and a cache-friendly `i-k-j`
  order.
- **Allocations:** Adds a large im2col buffer. For Conv2, the matrix is roughly
  `[288, 12,544]`, or about `3.6 million` `f64` values (~29 MiB), plus gradients
  and the GEMM workspace.
- **Cache locality:** Potentially excellent inside a well-blocked GEMM, but the
  im2col construction and col2im scatter add extra passes over large buffers.
- **Pros:** Makes standard GEMM optimizations applicable; potentially large
  gains if a good custom GEMM kernel is written.
- **Cons:** Extra memory traffic and complexity; a scalar hand-written GEMM may
  be slower than the current direct kernel. Requires careful handling of
  padding, stride, dilation, and overlapping col2im accumulation.

### 6.7 Native BLAS

**Idea:** Use im2col or a native convolution interface backed by an optimized
BLAS/library implementation.

- **Data access:** Same matrix transformation as custom GEMM, but the library
  chooses blocking, SIMD, and possibly multithreading.
- **Allocations:** Still needs im2col unless the library exposes a direct
  convolution API; library workspace allocations are implementation-specific.
- **Cache locality:** Usually very good because vendor GEMM kernels are tuned
  for the target CPU.
- **Pros:** Potentially the largest single-step performance improvement.
- **Cons:** New dependency, portability and licensing considerations, possible
  thread oversubscription, and a significant architectural decision. It must
  not be introduced without approval.

### 6.8 Explicit SIMD or improved autovectorization

**Idea:** Make the contiguous `CO` loops explicitly vectorizable or use platform
intrinsics.

- **Data access:** No layout change; operate on the existing contiguous packed
  vectors.
- **Allocations:** None beyond the existing buffers.
- **Cache locality:** Same as current, but fewer scalar loop iterations and
  better instruction-level parallelism.
- **Pros:** Low memory overhead and no im2col buffer. Potentially useful for
  both forward and backward.
- **Cons:** `f64` SIMD width is smaller than `f32`; explicit SIMD is
  architecture-specific; the compiler may already autovectorize the direct
  `zip` loops. Manual intrinsics would reduce portability.

### 6.9 Kernel-level multithreading

**Idea:** Partition batch, output-channel, or spatial-tile work across CPU
cores.

- **Data access:** Each thread works on a disjoint tile. Input/weight buffers
  can be shared read-only.
- **Allocations:** `grad_input` needs per-thread accumulation buffers followed
  by a reduction, or a synchronization strategy. `grad_weight` similarly needs
  careful reduction.
- **Cache locality:** Each core gets a smaller working set, but shared-memory
  bandwidth and synchronization can offset the gain.
- **Pros:** Uses all CPU cores and can scale better than single-thread SIMD for
  large tensors.
- **Cons:** The autograd graph uses `Rc` and is not currently designed for
  cross-thread execution. Thread-pool overhead, reduction cost, oversubscription
  with a threaded BLAS, and nondeterministic accumulation order are concerns.
  Correctness work is substantial.

### 6.10 Fused bias and small elementwise operations

**Idea:** Add bias during the convolution output write or fuse nearby ReLU and
broadcast operations.

- **Data access:** Removes a separate pass over the output tensor.
- **Allocations:** Usually removes temporary broadcast/bias graph buffers.
- **Cache locality:** Avoids rereading and rewriting large activation tensors.
- **Pros:** Low algorithmic risk and useful for inference.
- **Cons:** The convolution arithmetic dominates, so the expected training gain
  is small. Bias fusion does not address the measured `grad_input` bottleneck.

## 7. Priority order

Based on the current profile, the practical order is:

1. Keep direct packed convolution as the baseline.
2. Try a small fixed spatial tile for `grad_input`.
3. Try a separate spatial tile for `grad_weight` only if profiling justifies it.
4. Consider pad-aware access or parameter-resident packed weights as
   allocation/cache experiments.
5. Consider im2col plus a custom GEMM only after direct tiling is evaluated.
6. Treat native BLAS, explicit SIMD, and multithreading as separate
   architectural/dependency decisions.
7. Do not reintroduce per-tap output-channel chunk blocking; it was measured to
   regress.

Every change should be evaluated with both:

```text
MINITORCH_PROFILE_LAYERS=true make run-release
make run-release
```

The first isolates layer and section timings; the second measures the clean
first-epoch-plus-validation result.
