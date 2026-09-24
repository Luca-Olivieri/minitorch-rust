# CNN Optimization Opportunities

This document records optimization work identified while comparing the Rust
`SmallCNN` implementation with the PyTorch CPU reference.

## Current bottlenecks

- `Conv2d` decomposes each kernel tap into slices, transposes, reshapes, and a
  2D matrix multiplication. A 3x3 convolution therefore creates many temporary
  tensors and autograd nodes.
- `TensorStorage::matmul` is a scalar triple loop without SIMD, blocking, or a
  native BLAS backend.
- Shape operations such as `slice_strided` and `reshape` frequently materialize
  fresh buffers, including for transposed views.
- `MaxPool2d` stores a `Vec<Vec<usize>>` for every output window. Large batches
  create hundreds of thousands of small heap allocations.
- The autograd graph uses `Rc`, boxed type-erased backward sources, and dynamic
  dispatch. This is useful for flexibility but adds overhead to every operation.
- Evaluation currently needs an explicit inference path so it does not build
  unnecessary autograd edges or pooling metadata.

## Highest-priority opportunities

1. **Direct convolution kernels**
   - Implement convolution as a direct nested-loop kernel over batch, output
     channel, output position, input channel, and kernel position.
   - Implement convolution weight, input, and bias gradients directly instead of
     differentiating through the tap decomposition.
   - Preserve the existing stride, dilation, padding, and bias semantics.

2. **Optimized matrix multiplication**
   - Add loop ordering, register blocking, and contiguous row/column access.
   - Explore a platform BLAS backend only if dependency and portability policy
     allows it.
   - Benchmark scalar, SIMD, and blocked implementations independently.

3. **Reduce intermediate allocations**
   - Fuse bias addition and ReLU where practical.
   - Avoid materializing transposed views before convolution or matrix
     multiplication.
   - Reuse temporary buffers where ownership and autograd semantics permit it.

4. **Compact max-pool metadata**
   - Replace `Vec<Vec<usize>>` with flat index storage plus offsets/counts, or a
     compact tied-max representation.
   - Keep tied-maximum gradient behavior unchanged.

5. **Explicit inference execution**
   - Add an `AutogradContext` so evaluation can disable autograd without global
     or thread-local state.
   - Use value-only pooling and avoid constructing backward sources during
     inference.

6. **Profiling and layer-level benchmarks**
   - Measure convolution, pooling, linear, activation, and autograd separately.
   - Track allocations and peak memory in addition to wall-clock time.
   - Keep CPU thread count, dtype, batch size, and input data fixed when
     comparing implementations.

## Lower-priority opportunities

- Cache frequently used shapes, strides, and broadcast metadata.
- Use a more compact tensor representation for contiguous tensors.
- Add fused optimizer/update kernels.
- Investigate allocator behavior and reduce short-lived `Vec` allocations.
- Add a dedicated batched tensor data path for dataset loading.
- Add optional native multithreading after single-thread correctness and
  profiling are established.

## Benchmark reference

The current CPU comparison uses `f64` and one Torch thread for a relatively
fair arithmetic comparison. PyTorch still benefits from optimized native CPU
kernels even with one thread. Benchmark results should therefore report:

- Rust forward, backward, and optimizer time separately.
- Rust and PyTorch training-step totals.
- Evaluation time with and without autograd.
- Dtype, device, thread count, batch size, and warm-up state.
