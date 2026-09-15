# Strong-Typed Dtype Plan

Reconstruction of the staged plan for strong-typed dtypes (bool, i8–i64,
u8–u64, f32, f64). Stages 0–4 are complete; Stage 5 (consumers) is the
remaining verification work. This file exists so the plan survives long
sessions (it was previously only recorded in conversation history).

## Guiding principles (locked)

- No type promotion: a tensor op is defined in exactly one family impl; mixing
  dtypes requires an explicit `cast` (`Tensor<bool> + Tensor<u8>` does not
  compile).
- Conversion legality = a cast-trait impl exists (compile-time, no runtime
  dispatch). Three tiers: `CastFrom` (exact, always on), `LossyCastFrom`
  (feature `allow_lossy_casts`), `DangerousCastFrom` (feature
  `allow_dangerous_casts`). The two features are independent.
- Autograd is generics-driven: `NBackwardOp<Op, N, T>` with `dyn
  GradFnTrait<T>`, gradients computed in `T` for `T: Float` (Option B, user
  choice).
- The `f64` path is byte-for-byte the pre-generic behavior. The 42 integration
  tests are the safety net.

## Stage 0 — Dtype layer (done)

Sealed `Dtype` family + arithmetic `Numeric` / `Float` / `Integer` sub-families
and the three cast tiers. Files: `src/core/dtype/{mod,casts}.rs`.
Determines op availability: `Dtype` (structural), `Numeric`, `Float` (IEEE:
f32/f64), `bool` implements only `Dtype`.

## Stage 1 — Generic plumbing (done)

- `TensorStorage<T: Dtype>`, `TensorNode<T>`, `FreeTensor<T>`/`GraphTensor<T>`
  with default `T = f64` (so all pre-existing code compiles unchanged).
- `AbstractTensor<T>` with `at() -> &T`, `shape`, `numel`, `requires_grad`,
  `item() -> T` (panic on non-singleton).
- Generic `IntoNestedStorage<T>` for `GraphTensor::wrap` + `DtypeStyler` for
  typed `Display`.

## Stage 2 — Cast methods (done)

`cast::<U>()` / `cast_lossy::<U>()` / `cast_dangerous::<U>()` on both tensor
flavors. Shape-preserving, no gradient edge.

## Stage 3 — Ops split by family (done, as internal `3a` storage + `3c` tensor)

Structural on `T: Dtype`; `impl<T: Numeric>` add/mul/matmul/maximum/sum/max/
`sub_scaled`/copy_d/copy_s/scalar `Add<T>`+`Mul<T>`; `impl<T: Float>` sub/div/
neg/pow/ln/exp/sqrt/mean/norm/dist/`is_close`/scalar `Sub<T>`+`Div<T>`; bool
land/lor/lnot. `argmax`/`one_hot` output `f64` regardless of input dtype
(`one_hot` via `T: OneHotLabel`).

## Stage 4 — Autograd, Option B (done, as internal `3b`)

- Option A (originally recommended): autograd f64-only.
- **Option B (chosen by user): autograd generic over `T: Float`.**

Stages 3+4 were merged into one push (`3a`–`3d`) after rustc confirmed E0592/
E0119: same-named ops cannot live in two family impls even with disjoint
bounds, so f64-only differentiable ops and generic int ops cannot coexist.

## Stage 5 — Consumers (done)

`nn/`, `data/` (no `io/` module exists yet) and the `main/` binaries stay f64
via the `GraphTensor = GraphTensor<f64>` default. Scope and outcome:

1. Confirm `nn/`, `data/`, `main/{xor,covertype}.rs` compile and behave
   unchanged under the f64 default — verified: `cargo check`, `clippy -D
   warnings`, and the full default test suite green (lib 40 pass/2 ignored,
   models 9, nn 1, tensor 53); both binaries build and `xor` trains.
2. Scalar-path ripple from `at()`/`item()` returning `&T`/`T` is a non-issue:
   with `T = f64` everything (loss-item smoothing, covertype averaging) stays
   `f64`.
3. Original plan predicted a comparison-return-type ripple through the tests;
   this is moot because comparisons keep same-dtype 1/0 masks.
4. Added end-to-end typed-boundary test `typed_labels_feed_f64_loss_pipeline`
   (tests/tensor/dtypes.rs): `i32` labels -> `one_hot` (int->f64 boundary) ->
   f64 cross-entropy-style loss + autograd.

## Deliberate divergences from the original plan

1. Comparisons (`gt/gte/lt/lte`) return **same-dtype 1/0 masks**
   (`T::ONE`/`T::ZERO`), not `Tensor<bool>` as the plan originally stated —
   preserves the f64 `1.0`/`0.0` behavior exactly.
2. `copy_d`'s `CopyDOp` gradient edge was dropped during Stage 1 (operand
   arrays were f64-typed) and **restored** once autograd went generic
   (`tensor/mod.rs`, tested by `copy_d_attaches_grad_edge`).
3. `mean`/`norm`/`dist` and scalar `Sub<T>`/`Div<T>` added during the merged
   push (Float-home).

## Follow-ups (documented in-repo, not formally staged)

- `SignedInteger`/`UnsignedInteger` split (`dtype/mod.rs`) to widen `neg`/`abs`
  (and differentiable `sub`/`div`) to signed ints.
- Differentiable casts (`tensor/cast.rs` currently "non-differentiable data
  transform").
- Graph-level `abs`/`modul` only exist at storage level; wire them onto tensors.
- `f16`/`bf16` future members (`dtype/mod.rs`).
- Run `cargo fmt` across the tree: `cargo fmt --check` currently reports
  pre-existing formatting debt in the Stage 0–4 files (not introduced by
  Stage 5); the `make check` gate includes `cargo fmt --check`.

See also `SESSION_CHANGES.md` (older roadmap: SIMD/threaded matmul, dead-code
cleanup).