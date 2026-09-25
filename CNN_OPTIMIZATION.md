# CNN Optimization Opportunities

This document tracks optimization work for the Rust `SmallCNN` implementation
relative to the PyTorch CPU reference. Benchmark results are maintained in
[`OPTIMIZATION_HISTORY.md`](OPTIMIZATION_HISTORY.md).

## Current bottlenecks

- The direct `Conv2d` kernels avoid the old per-tap graph, but their scalar
  inner loops still need profiling and further blocking/vectorization work.
- `TensorStorage::matmul` remains a scalar triple loop without SIMD, register
  blocking, or a native BLAS backend. It is still used by `Linear` and other
  matrix operations.
- Shape operations such as `slice_strided` and `reshape` can still materialize
  fresh buffers in other layers and utility paths.
- `MaxPool2d` now avoids metadata allocation on its value-only/no-grad path and
  stores training maxima in flat index/offset buffers. Further gains depend on
  measuring the pooling kernels separately from convolution.
- The autograd graph uses `Rc`, boxed type-erased backward sources, and dynamic
  dispatch. This is flexible but adds overhead to differentiable operations.
- Evaluation now accepts an explicit `no_grad` flag, but it still uses the
  `GraphTensor` storage path rather than a node-free raw-tensor path.
- The current direct convolution backward is first-order only and is now the
  main training-time optimization target after the forward kernel improvement.

## Completed optimizations

- [x] Replace separate `requires_grad`, `no_grad`, and `grad_fn` state with
      `Option<AutogradMeta>` (`Leaf` and `Node`).
- [x] Add explicit `no_grad: bool` propagation through `Forward1` and tensor
      operations without global or thread-local state.
- [x] Add a direct groups-one `Conv2d` forward storage kernel.
- [x] Add direct first-order `Conv2d` input and weight gradients.
- [x] Avoid materializing the old per-tap convolution graph.
- [x] Pack convolution weights for contiguous output-channel access.
- [x] Reuse input values across output channels in the forward kernel.
- [x] Reuse packed output-gradient and weight tensors in the backward kernel.
- [x] Add a dedicated `Conv2dOp` autograd rule.
- [x] Add a true value-only `MaxPool2d` path that skips maximum metadata.
- [x] Replace per-window `Vec<Vec<usize>>` pooling metadata with flat indices and
      offsets while preserving tied-maximum gradient splitting.
- [x] Keep bias as a separate operation for now; bias fusion is deferred.

## Highest-priority optimizations still to check

### 1. Benchmark and profile the optimized convolution

- [ ] Run a complete release benchmark with the same CPU, dtype, and thread
      settings used for the PyTorch reference.
- [ ] Record forward, loss, backward, and optimizer separately.
- [ ] Record complete epoch and validation times, not only initial evaluation.
- [ ] Profile the two convolution layers independently.
- [ ] Check whether packing or padded-input construction is a meaningful part
      of the remaining runtime.
- [ ] Measure peak memory and short-lived allocations.

The latest user-provided result reduced step-100 backward time to approximately
`214.6 ms`, but complete-epoch and validation measurements are still needed.

### 2. Bias fusion

- [ ] Fuse bias initialization/accumulation into the direct convolution
      forward kernel.
- [ ] Compute bias gradients in the convolution backward rule.
- [ ] Measure the change separately from the direct-kernel optimization.
- [ ] Preserve the existing parameter layout and optimizer behavior.

### 3. Optimized matrix multiplication

- [ ] Replace the scalar `i-k-j` loop with a cache-friendly loop order.
- [ ] Add register blocking for the `Linear` layers.
- [ ] Explore contiguous row/column access patterns.
- [ ] Benchmark scalar, manually blocked, and any approved native backend
      implementations independently.
- [ ] Keep the implementation portable and avoid unapproved dependencies.

### 4. Reduce intermediate allocations

- [ ] Reuse temporary buffers where ownership and autograd semantics permit.
- [ ] Avoid materializing transposed or reshaped views before matrix operations.
- [ ] Investigate whether the remaining `GraphTensor` node allocations matter
      after convolution is optimized.
- [ ] Consider a separate raw-storage inference path only if profiling shows a
      meaningful benefit.

### 5. Compact max-pool metadata

- [x] Replace `Vec<Vec<usize>>` with flat index storage plus offsets.
- [x] Preserve even gradient splitting for tied maxima.
- [x] Make the value-only/no-grad path avoid maximum metadata allocation.
- [ ] Benchmark the value-only path separately from training backward.
- [ ] Consider a bitmask or backward recomputation only if profiling shows the
      flat representation is still a meaningful cost.

### 6. Layer-level profiling and benchmarking

- [ ] Measure convolution, pooling, linear, activation, and autograd separately.
- [ ] Track allocations and peak memory in addition to wall-clock time.
- [ ] Keep CPU thread count, dtype, batch size, input data, and warm-up state
      fixed when comparing implementations.
- [ ] Record commit/build information with every benchmark result.

## Lower-priority opportunities

- Cache frequently used shapes, strides, and broadcast metadata.
- Use a more compact representation for contiguous tensors.
- Add fused optimizer/update kernels.
- Investigate allocator behavior and reduce short-lived `Vec` allocations.
- Add a dedicated batched tensor data path for dataset loading.
- Investigate optional native multithreading after single-thread correctness and
  profiling are established.

## Benchmark reference

The current CPU comparison uses `f64` and one Torch thread for a relatively fair
arithmetic comparison. PyTorch still benefits from optimized native CPU kernels
even with one thread. Benchmark results should therefore report:

- Rust forward, backward, and optimizer time separately.
- Rust and PyTorch training-step totals.
- Evaluation time with and without autograd.
- Dtype, device, thread count, batch size, and warm-up state.
