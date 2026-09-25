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

### 1.1 Measured baseline

The current code is the post-revert state with the **stride-aware matmul**
rewrite, the **register-blocked conv accumulator**, and the **position-tiled
weight gradient** applied. The per-tap output-channel chunk blocking (§5.1) and
the square-kernel specialization (§5.2) both remain removed.

Captured on the development machine: Apple M1 MacBook Air, 8 GB, 4P + 4E cores,
`rustc 1.97.1`, release profile with `lto = true` and `codegen-units = 1`.

| Metric | Value |
|---|---:|
| Epoch-1 training time | `229.711 s` |
| Initial loss evaluation | `10.407 s` |
| Validation loss time | `10.493 s` |
| Smoothed training loss | `0.33636283742747686` |
| Validation loss | `0.26916314672841835` |

Training time fell `2.2%` from `234.815 s` on the previous profiled run, and
`23.3%` from the `299.386 s` clean baseline recorded in Stage 6.

**The loss values are no longer bit-identical to Stages 0–8.** The position tile
accumulates eight positions into a register partial before flushing, where the
original kernel left-folded all positions in sequence. Visiting positions in the
same order is not sufficient for bit-identity; the grouping into partials is
itself a reassociation, the same effect as §6.3. The observed shift is one unit
in the last place. The bit-identity gate is no longer available as a
correctness check; see §1.4.

Steady-state section timings, taken as the median over the nine logged steps
100–900. Step 938 is a half batch and is excluded; see §1.2.

| Conv2 backward | Median | Conv1 backward | Median |
|---|---:|---|---:|
| `pack grad_output` | `1.95 ms` | `pack grad_output` | `3.28 ms` |
| `pack weight` | `0.02 ms` | `pack weight` | `0.00 ms` |
| `padded input` | `2.00 ms` | `padded input` | `0.24 ms` |
| `grad_weight` | `45.39 ms` | `grad_weight` | `3.04 ms` |
| `grad_input` | `51.85 ms` | `grad_input` | `3.27 ms` |
| `unpack grad_weight` | `0.02 ms` | `unpack grad_weight` | `0.00 ms` |
| **total** | **`101.36 ms`** | **total** | **`10.03 ms`** |

| Forward | Median | Backward op | Median |
|---|---:|---|---:|
| `conv1` | `9.53 ms` | `conv2d` | `111.40 ms` |
| `conv2` | `41.66 ms` | `matmul` | `24.37 ms` |
| `linear1` | `3.57 ms` | `maximum` | `20.23 ms` |
| `pool1` | `7.39 ms` | `max_pool2d` | `4.14 ms` |
| `relu1` | `4.78 ms` | `add` | `2.55 ms` |
| `pool2` | `4.04 ms` | `mul` | `0.58 ms` |
| `relu2` | `2.37 ms` | | |
| `dropout` | `4.88 ms` | | |

Per-step totals: forward `~79 ms`, backward `~165 ms`.

### 1.2 Two diagnostic observations

**Every section scales linearly with batch size.** Step 938 is a half batch and
lands at `49.4–49.9%` of the full-batch median in every section: `grad_input`
`32.2 ms`, `grad_weight` `25.3 ms`, `conv2` forward `29.1 ms`, `matmul`
`21.4 ms`. The kernels are therefore pure steady-state throughput work with
negligible fixed cost. There is no first-touch page-fault or warmup artefact
inflating the numbers, and no memory-footprint cliff.

**Conv1's `pack grad_output` is its single largest section.** At `3.27 ms` it
exceeds Conv2's `1.92 ms` despite Conv1 doing far less arithmetic, because
Conv1's output is `[64, 32, 28, 28]` = 1.6 M elements against Conv2's 0.8 M.
Packing is roughly 34% of Conv1's entire backward pass. This is the clearest
case in the model where removing a data transformation beats optimizing a
kernel, and it is invisible in the aggregate `conv2d` total.

### 1.3 Arithmetic intensity
Conv2 performs `462.4 MFLOP` in each of its three main loops, which is
`231,211,008` multiply-accumulates either way: output-stationary it is
`64 × 14 × 14` output positions times `32 × 9 × 64` channels and taps;
input-stationary it is `64 × 32 × 14 × 14` input positions times `9 × 64` taps
and output channels. M1 has 128-bit datapaths only, so two FMA-capable pipes
with two `f64` lanes each give 8 FP64 flops/cycle, about `25.6 GFLOP/s` at
3.2 GHz.

| Loop | Time | Achieved | Share of ~25.6 GFLOP/s |
|---|---:|---:|---:|
| `grad_input` | 51.85 ms | 8.9 GFLOP/s | ~35% |
| `grad_weight` | 45.39 ms | 10.2 GFLOP/s | ~40% |
| forward | 41.66 ms | 11.1 GFLOP/s | ~43% |
| `conv1` forward | 9.53 ms | 3.0 GFLOP/s | ~12% |

The two tiling changes lifted all three loops, and the loops are now
converging rather than separating: `35%`, `40%`, `43%`. The remaining gap is no
longer dominated by accumulator traffic in any of them; see §4.5 and §6.4 for
what is left in the inner loops.

Two caveats make these figures *favourable* to any argument that the code is
vectorization-starved. The process is single-threaded on a 4P + 4E machine, so
it migrates between Firestorm performance and efficiency cores and the true
P-core share is likely 35–45%. And section-to-section run-to-run variance of up
to 15% is scheduler noise.

Even on the favourable reading, all three Conv2 loops land in the same narrow
band around a third of peak. If autovectorization were failing in one loop it
would be an outlier, not a uniform third. §4.5 identifies what they share.

Conv1's forward pass is the outlier at ~10% of peak, and for a different
reason than Conv2: its output is written with a plane-strided pattern, one
8-byte element per `28 × 28`-element stride, so the write stream defeats
prefetch. It has far less arithmetic, so a 10× inefficiency still only costs
`9.5 ms`.

### 1.4 The bit-identity gate is closed

Stages 0 through 8 all report a smoothed training loss of
`0.3363628374274767` and a validation loss of `0.2691631467284183`. Stage 9
reports `0.33636283742747686` and `0.26916314672841835`.

The cause is understood and benign: two kernels now reassociate. The Stage 8
`grad_input` reduction sums eight-element output-channel block partials rather
than folding 64 channels sequentially. The Stage 9 weight gradient accumulates
eight positions into a register partial before flushing, rather than
left-folding every position in sequence. **Visiting values in the same order is
not sufficient for bit-identity — the grouping into partials is itself a
reassociation.** I got this wrong when writing the Stage 9 change, having
already made the identical observation in Stage 8.

Consequences for the workflow:

- Loss comparison is no longer an exact gate. It becomes a magnitude check: a
  shift of a unit in the last place is expected, anything larger is not.
- Correctness now rests on `tests/nn/conv2d.rs`, which compares forward,
  `grad_input`, and `grad_weight` against independent triple-loop reference
  implementations at `1e-9` tolerance, and which has mutation-tested coverage of
  every tiling body and every remainder path.
- Any further reassociation should be recorded in this section, not treated as
  a regression.

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

- The inner output-channel loop is contiguous, and §4.5 confirms the compiler
  vectorizes it to the full `v.2d` width.
- The accumulator is small and reused.
- Each input value is reused across all output channels.
- Packed weights avoid repeated strided reads of the original weight layout.

Costs:

- Packing weights adds a read/write pass over the parameter buffer. Measured at
  `0.02 ms` for Conv2, so this is negligible in practice.
- Padding copies the input before the arithmetic pass, measured at `2.06 ms`
  for Conv2.
- The output write is strided across channel planes. This is what holds Conv1
  forward to ~10% of peak in §1.3.
- The direct loop has no explicit spatial or channel tiling.
- **The accumulator is reloaded and rewritten once per tap**, which §4.5
  measures at 50% of the forward loop's memory traffic. This is the single
  largest cost in the forward pass and is invisible from the source, because
  the accumulator is only 512 bytes and looks entirely cache-resident.

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
- **The packed weight-gradient accumulator is rewritten once per output
  position**, not once per batch. §4.5 measures this at 67% of the loop's
  memory traffic, the worst share of the three conv loops, which is why
  `grad_weight` is treated as a separate step in §7 rather than assumed to
  follow `grad_input`.

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

This is the section that dominates Conv2 backward, level with `grad_weight` at
`51.85 ms` and `45.39 ms` respectively — together 96% of the pass.

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

### 4.5 Instruction-level analysis of the hot loops

The following was read out of the release binary (`llvm-objdump -d` over
`conv2d_backward_impl::<f64>` and over
`<Conv2d as Forward1>::forward`, both on `aarch64-apple-darwin`). It replaces
speculation with measurement.

All three conv inner loops — forward, `grad_weight`, and `grad_input` — **are**
vectorized, and to the hardware maximum. A 128-bit datapath holds two `f64`
lanes, so `v.2d` at two lanes is the widest `f64` vector this CPU can express.
Each loop is unrolled to four accumulators and processes 8 lanes per iteration.
The `grad_input` loop:

```text
loop:                             ; 8 f64 lanes per iteration
  ldp  q0, q1, [x3]        ldp  q2, q3, [x3, #0x20]   ; 8 lanes of grad_output
  ldp  q4, q5, [x0,#-0x20] ldp  q6, q7, [x0], #0x40   ; 8 lanes of packed_weight
  fmul.2d v0..v3
  ldp  q4, q5, [x1,#-0x20] ldp  q6, q7, [x1]          ; 8 lanes of accumulator
  fadd.2d v0..v3
  stp  q0, q1, [x1,#-0x20] stp  q2, q3, [x1], #0x40   ; 8 lanes of accumulator
```

Forward and `grad_weight` share the same shape, with one operand broadcast as
a scalar lane instead of loaded as a vector:

```text
loop:                             ; 8 f64 lanes per iteration
  ldp  q1, q2, [x24,#-0x20] ldp  q3, q4, [x24], #0x40 ; 8 lanes of packed_weight
  fmul.2d v1..v4, v0[0]                                 ; scalar broadcast operand
  ldp  q5, q6, [x25,#-0x20] ldp  q7, v16, [x25]        ; 8 lanes of accumulator
  fadd.2d v1..v4
  stp  q1, q2, [x25,#-0x20] stp  q3, q4, [x25], #0x40 ; 8 lanes of accumulator
```

Per 8 lanes:

| Loop | loads | stores | total mem ops | FP ops | Accumulator share |
|---|---:|---:|---:|---:|---:|
| forward | 12 | 4 | 16 | 8 | 8/16 = 50% |
| `grad_weight` | 8 | 4 | 12 | 8 | 8/12 = 67% |
| `grad_input` | 12 | 4 | 16 | 8 | 8/16 = 50% |

Two facts follow.

**The accumulator round-trip is 50–67% of all memory traffic, and it is pure
overhead.** The same 512-byte accumulator is loaded and stored once per tap —
9 times per output position — when it only needs to be touched once. That
traffic is the dominant cost, and the data it carries is the same data every
time. The root cause is that `out_channels` is a runtime value, so the
accumulator is a `Vec<T>` behind a `&mut [T]`: LLVM cannot scalar-replace it
into registers, because the trip count is unknown.

**There is no `fmla` anywhere in the binary.** Rust does not permit
floating-point contraction, so `accum += a * b` compiles to a separate `fmul.2d`
and `fadd.2d` rather than one fused instruction. `f64::mul_add` is the only way
to get the fused form; it is a std method, stable, and adds no dependency.

Arithmetic intensity confirms the diagnosis. `grad_input` at `65.06 ms` and an
estimated 3.2 GHz implies about 518 cycles per output position, carrying 576
vector loads and 288 vector stores. Two 128-bit loads per cycle gives a
floor of roughly 432 cycles, so the loop sat at ~84% of its load-port limit
while using only about 28% of the FMA throughput. **The loops were load-port
bound, not vector-width bound.** This is why adding SIMD cannot help, and why
reducing loads is the only lever that matters.

The load-port floor was itself too pessimistic. `grad_input` now runs at
`51.85 ms`, which is 35% of peak rather than the 84%-of-load-limit the model
implied, so the residual is not purely memory-side. Two bounds-check branches
and a stack reload of the tile length remain per iteration, and the block loop
recomputes tap addresses eight times per position. See §7 item 4.

**Status: fixed, and the prediction was too optimistic.** §6.3 has since been
applied to the forward loop and to `grad_input`. Both now emit the accumulator
as `v.2d` registers held across the tap loop, with no accumulator memory
traffic at all:

```text
grad_input, one tap, 8 output channels:
  ldp q7, q6, [x1, #0x20]   ldp q17, q16, [x1]      ; grad_output
  ldp q19, q18, [x21, #0x20] ldp q21, v20, [x21]   ; packed_weight
  fmul.2d v17, v16, v7, v6
  fadd.2d v3, v3, v6        ; accumulators live in v2..v5
  fadd.2d v4, v4, v7
  fadd.2d v5, v5, v16
  fadd.2d v2, v2, v17
```

Memory operations per 8 lanes per tap fell from 12 loads + 4 stores to 8 loads
and 0 stores, and all 8 remaining loads are useful. The forward loop improved
the same way: 8 memory ops down to 3.

Measured effect: Conv2 `grad_input` `65.06 → 51.16 ms` (−21.4%), Conv2 forward
`59.15 → 41.65 ms` (−29.6%), Conv1 `grad_input` `4.58 → 3.25 ms` (−29.0%).
The load-port model predicted `grad_input` would roughly halve; it did not.
Three costs remain in the inner loop that the memory model ignored:

- Two bounds-check branches per iteration, from the sub-slice range checks.
- A stack reload of the tile length (`ldr x22, [sp, #0x200]`) inside the loop.
- The block loop now sits outside the tap loop, so tap addresses are recomputed
  eight times per output position instead of once.

**The arithmetic-intensity models in this document are useful for ruling a
change out, not for sizing it.** Two predictions have now come in roughly 2×
optimistic: the matmul rewrite (§4.6) and this one.

Because the final reduction over output channels is now a sum of eight-element
block partials rather than a 64-element sequential fold, `grad_input`
**reassociates**. Forward and `grad_weight` are bit-identical, since neither
reorders its own accumulation. Despite the structural reassociation, the
measured loss values are bit-identical to every prior profile — the
reassociated sums evidently produce the same doubles for this data.

### 4.6 The same defect, fixed, in `matmul`

The matmul rewrite validated this analysis before it was applied to convolution.
`TensorStorage::matmul` had **zero** vectorized `f64` operations: the whole
binary contained 24 `fmul.2d`, all inside the two conv functions, and matmul
contributed only scalar `fmul d` / `fadd d`. The cause was that the inner loop
indexed `b` through a runtime stride, so the compiler could not prove the
access was contiguous and emitted no runtime versioning guard. `dL/dA =
grad @ W^T` made it worse, walking every output column at a full row stride.

The fix mirrors §6.3 exactly — hold the partial sums in a fixed-size stack tile
so they stay in registers — and the resulting disassembly confirms the
predicted codegen:

```text
loop:                             ; one k step, 8 output columns
  ldr  d4, [x17, x0, lsl #3]      ; a_val, scalar
  ldp  q5, q6, [x5, #-0x20]       ; 4 f64 of b, contiguous
  fmul.2d v5, v5, v4[0]
  fadd.2d v0, v0, v5              ; accumulators live in v0..v3
  fmul.2d v5, v6, v4[0]
  fadd.2d v1, v1, v5
  ldp  q5, q6, [x5]
  fmul.2d v5, v5, v4[0]
  fadd.2d v2, v2, v5
  fmul.2d v4, v6, v4[0]
  fadd.2d v3, v3, v4
```

| per 8 columns × 1 `k` | before | after |
|---|---:|---:|
| memory ops | 24 | 3 |
| accumulator load/store | 16 | 0 |
| arithmetic | scalar | `v.2d`, 2 lanes |

Measured effect: `matmul` backward `42.95 → 24.61 ms` and `linear1` forward
`16.54 → 3.58 ms`. An intermediate version vectorized but emitted eight bounds
checks per `k` step, because the compiler could not prove `b_row_base + u` was
in range; taking a fixed-length sub-slice replaced those with a single range
check and let the full tile vectorize.

The lesson transfers directly to §6.3: in both kernels the accumulator traffic
dominated, and in both the fix was a compile-time-sized stack array, not a
spatial or channel reordering.

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
  serialization. Expected gain is small: the measured pack-weight section is
  `0.02 ms` for Conv2 and below the timer's resolution for Conv1, so this
  cannot return more than a rounding error.

  **Status: confirmed negligible, do not pursue.**

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
  can hurt vectorization if the compiler cannot hoist the boundary checks —
  which matters more than usual here, because §4.5 shows the loops already sit
  at ~84% of their load-port limit and any extra work is expensive. The
  measured padded-input section is `2.06 ms` for Conv2 and `0.24 ms` for
  Conv1, so the ceiling is roughly 2.3 ms per step for a full removal. Note
  the forward loop's own padding cost is not broken out separately, so the
  total is higher than the backward figure alone suggests.

  **Status: moderate, but blocked behind §6.3.** Any extra per-tap work is
  expensive while the loops sit at ~84% of their load-port limit.

### 6.3 Keep the accumulator in registers (highest priority)

**Idea:** Replace the heap accumulator with a compile-time-sized stack tile and
make the tap loop the innermost dimension over that tile, so the accumulator
lives in vector registers instead of being round-tripped to memory once per
tap. A residual tail slice handles `out_channels` that is not a multiple of
the tile width.

This is a refinement of the original "spatial tiling" idea, and the section
timings plus the disassembly in §4.5 show that **the accumulator, not the
spatial dimension, is the thing to tile.** Spatial blocking shares weight
loads, which is the smaller half of the traffic; register-blocking the
accumulator removes the larger half.

```text
current:  for position:  for tap:  acc[0..CO] += dy[..] * w[..]   ; reload per tap
proposed: for position:  for lane_block:                           ; CO / BLOCK blocks
                             acc[BLOCK] = 0                        ; registers
                             for tap:  acc += dy[..] * w[..]       ; held in registers
                             grad_input[..] += acc
```

- **Data access:** Removes 8 of the 16 memory operations per 8 lanes in
  `grad_input` and 8 of 12 in `grad_weight`, because the accumulator is
  written once per position instead of once per tap.
- **Allocations:** A fixed-size stack array of the chosen tile width, plus a
  `Vec` for the non-multiple tail. No heap traffic in the common case, and
  nothing anywhere near im2col scale.
- **Cache locality:** Neutral to mildly positive. This is a register-locality
  win, not a cache win — that is the point, because §4.5 shows the loops are
  load-port bound rather than cache or FP bound.
- **Pros:** No new dependency, no `unsafe`, no platform-specific code, keeps
  the generic stride/dilation/padding paths intact, and can be applied to
  forward, `grad_weight`, and `grad_input` with the same shape. It works
  precisely because the inner `CO` loop is already vectorized, so a
  fixed-size stack array is exactly what LLVM's SROA needs to keep the
  accumulators in `v.2d` registers. Expected gain from the load-port
  arithmetic is roughly 1.8×, putting `grad_input` near `36 ms` and the whole
  Conv2 backward near `90 ms`.
- **Cons:** Register pressure rises; a tile that is too wide spills and is
  worse than the status quo, so the width needs measuring. The tap loop
  becomes innermost, which forces a tail branch and makes boundary handling
  slightly more awkward. Contraction of the accumulator into registers also
  reassociates the tap sum, which this project permits but which will move the
  loss values in their last digits. Critically, §5.1 shows that a
  per-tap `chunks_mut` version of this idea regressed badly — the tile must be
  a fixed-size array indexed directly, not dynamic chunk iterators.

  **Status: applied to forward and `grad_input`, measured.** See §4.5. The tile
  width is `OUT_CHANNEL_TILE = 8` in `conv.rs`, chosen to match the four `v.2d`
  accumulators the compiler already selects, with a scalar remainder loop for
  `out_channels` that is not a multiple of it. Delivered `−21.4%` on `grad_input`
  and `−29.6%` on forward, against a predicted `~2×`. The `grad_input` reduction
  over output channels now sums eight-element block partials instead of folding
  64 elements sequentially, so it **reassociates**; forward and `grad_weight`
  remain bit-identical, and the measured losses did not move.

### 6.4 Position-tiling `grad_weight` — applied and measured

**`grad_weight` cannot be register-blocked the way §6.3 did the others.** Its
accumulator is read-modify-written once per *output position* — 12,544 times per
tap for Conv2 — so the value has to survive the entire position loop. But a
register tile of *positions* works: for one channel block, accumulate eight
positions in registers and flush once per tile.

- **Data access:** Memory operations per 8 positions × 8 channels × 1 tap fell
  from 96 to 48, and the accumulator's share of the inner loop went from 64
  operations to zero. Verified in the release binary:

```text
loop over 8 positions, one channel block, one tap:
  ldr  d22, [x13, x22, lsl #3]      ; input_value
  ldp  q23, q24, [x8]               ; 8 f64 of packed_grad_output
  fmul.2d v23, v23, v22[0]
  fadd.2d v20, v20, v23             ; tile in v18..v21 across the whole loop
  fmul.2d v23, v24, v22[0]
  fadd.2d v21, v21, v23
  ldp  q23, q24, [x8, #0x20]
  fmul.2d v23, v23, v22[0]
  fadd.2d v18, v18, v23
  fmul.2d v22, v24, v22[0]
  fadd.2d v19, v19, v22
  b.ne loop

flush, once per tile:
  ldp  q22, q23, [x8] → fadd → stp → ldp → fadd → stp
```

- **Cache locality:** Tiles also shorten the packed-gradient reuse distance from
  the whole 144 KiB buffer to `kernel_w * out_channels` — 1.5 KiB for Conv2 —
  which keeps it L1-resident. A plain loop interchange would have improved the
  same reuse distance but at the cost of re-streaming all 6.13 MiB of
  `grad_output` once per tap; the position tile keeps `grad_output` L1-reused
  because a position's 512-byte slice is still touched by consecutive taps.
- **Allocations:** Unchanged. The two remainders — positions when the output
  plane is not a multiple of `POSITION_TILE`, and channels when `out_channels`
  is not a multiple of `OUT_CHANNEL_TILE` — use a simple scalar loop.
- **Pros:** Largest convolution section, structural rather than a tuning race,
  no new data structures, and the change is isolated: every untouched control
  held within 1.7% across the profile.
- **Cons:** A first attempt that tabulated the tile's row and column into two
  `[usize; 8]` arrays spilled both the indices and `input_values` to the stack,
  and inflated the vector-FMA count in the function from 32 to 124. Recomputing
  row and column at each use removed the spill entirely: zero `ldr q, [sp]`
  remain and the count returned to 32. Register pressure, not memory traffic, is
  the binding constraint here, which is the opposite of §4.5.
- **Measured:** Conv2 `grad_weight` `51.07 → 45.39 ms` (−11.1%), Conv2 backward
  total `106.72 → 101.36 ms`, epoch-1 training `234.82 → 229.71 s`. Predicted
  `30–40 ms`; delivered `45.39 ms`. **Conv1 `grad_weight` regressed 19.7%**,
  from `2.54` to `3.04 ms`, and both distributions are tight enough that this is
  real rather than noise. Conv1 has one input channel against Conv2's 32, so it
  has nine taps instead of 288 and the per-tile setup is amortised over roughly a
  thirtieth of the arithmetic. Net effect is `−5.68 + 0.50 = −5.18 ms` per step.
- **Open:** The Conv1 regression is worth `0.50 ms` per step and would be
  recovered by skipping tiling when `in_channels * out_channels` is small. That
  wants a structural predicate rather than a bare constant, and it should not be
  done without deciding what the predicate is.
- **Tuning knob:** `POSITION_TILE` is 8. Halving it to 4 would halve the live
  `input_values` state at the cost of flushing twice as often. Given both the
  spill history and the Conv1 regression, that is worth an A/B rather than a
  guess.
- **Status: applied, measured, one open item (the size guard).**

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

### 6.8 Explicit SIMD, `mul_add`, and improved autovectorization

**Status: the intrinsics half is dead. The `mul_add` half is cheap but not the
bottleneck.** Analyzed against the release binary on the development machine
(Apple M1, `aarch64-apple-darwin`, `rustc 1.97.1`).

Verified properties of the target:

- `neon` is **already enabled by default**. `rustc --print cfg` and
  `rustc -C target-cpu=native --print cfg` produce byte-identical output, so
  there is no target feature to add. AArch64 FMA is part of the base FP/NEON
  ISA and is not feature-gated.
- The release profile already sets `opt-level = 3`, `lto = true`, and
  `codegen-units = 1`, so the usual autovectorization knobs are maxed.
- `std::simd` is `#[unstable(feature = "portable_simd", issue = "86656")]` in
  this toolchain's `std/src/lib.rs`. It needs a pinned nightly or
  `RUSTC_BOOTSTRAP=1`.
- The `core::arch::aarch64` `f64` intrinsics (`vld1q_f64`, `vst1q_f64`,
  `vfmaq_f64`, `vaddvq_f64`, `vgetq_lane_f64`, `vgetq_high_f64`) are all
  `#[stable(feature = "neon_intrinsics", since = "1.59.0")]`. No crate needed.

**The ISA ceiling.** M1 has 128-bit datapaths only: no AVX, no AVX2, no
AVX-512, no x86 FMA. A `f64` register holds two lanes, so the `v.2d` the
compiler already emits is the widest `f64` vector this CPU can express. Two
FMA-capable pipes give 8 FP64 flops/cycle, roughly 25.6 GFLOP/s at 3.2 GHz.
Because this project is `f64` to match the PyTorch reference, it sits in the
worst possible lane-count regime, where explicit SIMD has the least headroom
rather than the most.

**Autovectorization is not the problem.** §4.5 shows the inner loops already
emit `v.2d` at full width, unrolled four accumulators deep. Hand-written
intrinsics would generate the same `fmla` the compiler already declines to emit,
and would not remove a single load. The measured arithmetic intensity puts the
loops at ~84% of their load-port limit and only ~28% of FMA throughput, so there
is no vectorization headroom to recover.

- **Data access:** No layout change for either variant.
- **Allocations:** None. Both are in-place rewrites of the existing
  accumulator update.
- **Cache locality:** Unchanged. Neither reduces memory traffic, which is the
  actual constraint.
- **Pros (`mul_add`):** Rust does not permit floating-point contraction, so
  `accum += a * b` emits `fmul.2d` followed by `fadd.2d`. `f64::mul_add` is a
  std method, stable since 1.11, zero dependencies, and emits a single `fmla`.
  It halves the FP instruction count and gives one rounding instead of two, so
  it is also strictly more accurate and moves results slightly closer to
  PyTorch, whose BLAS kernels use FMA. In the `grad_weight` and forward loops
  the broadcast-multiply shape may make it more valuable than it looks.
- **Cons (`mul_add`):** The loops are load-port bound, not FP-issue bound, so
  the expected gain in `grad_input` is small — bounded by the ~28% FMA
  utilization. It changes results, so it must land as its own measured step
  rather than bundled with §6.3. It must not be smuggled in as
  `unsafe { vfmaq_f64(..) }`; the std method gets the same instruction with
  portable, safe code.
- **Pros (intrinsics):** Effectively none that `mul_add` plus the existing
  autovectorization do not already cover, except control over unrolling.
- **Cons (intrinsics):** Requires `#[cfg(target_arch = "aarch64")]` plus a
  scalar fallback or the crate stops building on x86 and Windows; introduces
  `unsafe` into kernels that are currently safe; adds no new instructions
  because the widths are already maximal; and does not address the 50–67% of
  memory traffic that is accumulator round-trips.
- **Dependencies:** None for `core::arch` or `mul_add`. `wide` and `packed_simd`
  are not justified: `wide`'s `f64x4` on a 128-bit machine is emulated as two
  2-lane operations, adding abstraction without adding instructions. Accelerate
  and vecLib are §6.7, not §6.8.

There is one place autovectorization genuinely cannot help, and it is a real
but bounded target. `grad_input` closes with
`input_accumulator.iter().fold(T::ZERO, |sum, &value| sum + value)`, 64
**dependent** `fadd`s, which LLVM will not vectorize or reorder because Rust
sets no `reassoc` or `fast` flags. The disassembly confirms it: 41 scalar
`fadd d` instructions and no `faddp`. `vaddvq_f64` reduces a two-lane vector in
a single `faddp`, and for two lanes that is `a[0] + a[1]`, bit-identical to the
scalar order; a pairwise tree is permitted in this project. At 25.7M adds
against a 65.06 ms section, though, this is a second-order term that §6.3
should be measured against first.

### 6.8.1 The lesson that transfers: make the stride provable, don't add instructions

§4.6 is the strongest evidence for this section's conclusion, because it is a
case where arithmetic genuinely *was* the gap and the fix still was not
intrinsics. `matmul` emitted no vectorized `f64` at all, purely because a
runtime stride left the compiler unable to prove its inner access contiguous.
The correction was to branch on `b_s1 == 1` so contiguity became a compile-time
fact within that path — no new instructions, no `unsafe`, no architecture
specialization. Convolution is not in that situation (§4.5: already `v.2d` at
full width), which is exactly why its remaining gap needs a different lever:
removing memory operations, not widening arithmetic.

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

Reordered against the measured baseline in §1.1 and the instruction-level
analysis in §4.5. §4.6 records the one item already delivered.

0. **Done: matmul stride-aware rewrite.** Applied and measured. `matmul`
   backward `42.95 → 24.33 ms`, `linear1` forward `16.54 → 3.57 ms`, losses
   bit-identical.
1. **Done: register-blocked conv accumulator (§6.3).** Applied to the forward
   loop and to `grad_input`, measured. Conv2 forward `59.15 → 41.65 ms`
   (−29.6%), Conv2 `grad_input` `65.06 → 51.16 ms` (−21.4%), losses
   bit-identical. Epoch-1 training `267.9 → 234.8 s`.
2. **Done: position-tiled `grad_weight` (§6.4).** Conv2 `grad_weight`
   `51.07 → 45.39 ms` (−11.1%), epoch-1 training `234.8 → 229.7 s`. Predicted
   `30–40 ms`, delivered `45.39 ms` — the third consecutive over-prediction, now
   a consistent factor of roughly two. One open item: Conv1's `grad_weight`
   regressed `2.54 → 3.04 ms` because a single-input-channel layer has nine taps
   and cannot amortise the per-tile setup. A size guard on
   `in_channels * out_channels` would recover `0.50 ms` per step, but it needs a
   structural predicate rather than a bare constant, and that is a decision to
   make explicitly.
3. **Target `max_pool2d_backward` (`reduce.rs`).** `20.23 ms`, stable across
   four profiles, third-largest backward category, and never attempted. The
   inner loop carries a bounds check with a `panic!` per scattered element and
   iterates the upstream gradient through a `strided_indices()` div/mod
   iterator. Unlike the tiling work its problem is structural rather than a tight
   optimization race, which makes it the cheapest unexplored code in the model.
4. **Remove the residual overhead in the tiled loops (§4.5).** `grad_input` is
   now the largest single convolution section at `51.85 ms`, and it still runs
   the pre-Stage-8 shape with two bounds-check branches and a stack reload of the
   tile length per iteration. It is also the one large convolution kernel with no
   measured improvement since Stage 8.
5. **Use `f64::mul_add` (§6.8).** Zero dependency and strictly more accurate.
   The tiled loops now run at 35–45% of FP64 peak, up from 28–35%, so there is
   less headroom in the FMA pipes than there was when this item was written.
   Still nearly free, but a smaller prize.
6. **Avoid materializing the padded input (§6.2).** `2.00 ms` for Conv2 and
   `0.24 ms` for Conv1. Worth less now that the arithmetic loops are no longer
   at ~84% of their load-port limit.
7. **Consider im2col plus a custom GEMM (§6.6) last.** The direct kernels are
   now at 35–45% of peak, which weakens the case considerably: im2col adds a
   ~29 MiB buffer and its own passes on top of loops that are no longer
   obviously inefficient.
8. **Treat native BLAS (§6.7) and multithreading (§6.9) as separate
   architectural/dependency decisions.** Neither is warranted before items 3–4
   are exhausted; both would multiply an already-serial pipeline rather than fix
   its arithmetic intensity.
9. **Do not reintroduce per-tap output-channel chunk blocking (§5.1);** it was
   measured to regress badly, and §6.3 is the structurally different
   replacement for the same idea.
10. **Do not pursue a parameter-resident packed weight (§6.1).** Measured at
    `0.02 ms` per call; it cannot return more than a rounding error.

Every change should be evaluated with both:

```text
make run-release
MINITORCH_PROFILE_LAYERS=true make run-release
```

The first measures the clean first-epoch-plus-validation result; the second
isolates layer and section timings. Note that a profiled run is not directly
comparable to a clean one, so compare like with like. Section deltas under
roughly 15% should be treated as scheduler noise, since the process is
single-threaded on a 4P + 4E machine and migrates between Firestorm
performance and efficiency cores; a step that regresses in every section at
once, as steps 200 and 500 did in the latest profile, is a thermal artifact
rather than a change in behaviour. A correctness change is never accepted on
timing alone: the smoothed training loss and validation loss must be compared
against the §1.1 baseline.

Every *untouched* path in a profile is a free control. In Stage 8, `grad_weight`
(−0.2%), `matmul` (−1.1%), `linear1` (−0.3%), `maximum` (−0.7%), and the
`pack grad_output` and `padded input` sections (−0.1% and +0.3%) all held
stationary, which is what established that the change was isolated. Compare
against those before believing a delta.
