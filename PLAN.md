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

Structural on `T: Dtype`; `impl<T: Numeric>` add/mul/sub/div/matmul/maximum/sum/max/
`sub_scaled`/copy_d/copy_s/scalar ops (`Add<T>`/`Mul<T>`/`Sub<T>`/`Div<T>`);
`impl<T: Signed>` neg; `impl<T: Float>` pow/ln/exp/sqrt/mean/norm/dist/
`is_close`; bool land/lor/lnot. `argmax`/`one_hot` output `f64` regardless of
input dtype (`one_hot` via `T: OneHotLabel`).

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
3. The original plan predicted a comparison-return-type ripple through the
   tests; it materialized and was resolved: comparisons now return `Tensor<
   bool>`, so the comparison + `is_close` call sites were updated to bool (and
   `maximum`/`max` backward reinterpret the mask via `as_numeric`).
4. Added end-to-end typed-boundary test `typed_labels_feed_f64_loss_pipeline`
   (tests/tensor/dtypes.rs): `i32` labels -> `one_hot` (int->f64 boundary) ->
   f64 cross-entropy-style loss + autograd.

## Deliberate divergences from the original plan (resolved)

1. ~~Comparisons (`gt/gte/lt/lte`) return **same-dtype 1/0 masks** (`T::ONE`/
   `T::ZERO`), not `Tensor<bool>` as the plan originally stated~~. **Resolved:
   comparisons now return `Tensor<bool>` as the plan specified.** The backward
   rules (`MaximumOp`/`MaxOp`) reinterpret the bool mask as 1/0 in `T` via an
   internal `as_numeric::<T>()` helper that bypasses the cast-trait table,
   keeping all backward rules purely `Numeric`-homed (no `CastFrom` bound leak
   into the forward `maximum`/`max` signatures).
2. ~~`sub`/`div`/`neg` were narrowed from `T: Numeric` to `T: Float`~~.
   **Resolved with a `Signed` family**: `Signed: Numeric + Neg` is implemented
   for `f32`/`f64` and `i8`…`i64`, and at the time `sub`/`div`/`neg` (tensor ops,
   scalar `Sub<T>`/`Div<T>`, and the `SubOp`/`DivOp`/`NegOp` grad rules) gated
   on it. `neg` still does (unsigned negation is undefined). Once backward
   dispatch was deferred (divergence 5), forward `sub`/`div` widened again to
   the `Numeric` home — unsigned ints now get them as forward-only ops, like
   PyTorch's uint arithmetic (no autograd). Float behavior is unchanged.
3. `copy_d`'s `CopyDOp` gradient edge was dropped during Stage 1 (operand
   arrays were f64-typed) and **restored** once autograd went generic
   (`tensor/mod.rs`, tested by `copy_d_attaches_grad_edge`).
4. `mean`/`norm`/`dist` added during the merged push (Float-home).
5. **Forward op availability was coupled to gradient-rule bounds.** Differentiable
   forward ops boxed `NBackwardOp<Op, N, T>` at construction, which requires
   `Op: GradRule<N, T>` — so an op's forward home leaked its backward math's
   bounds and `maximum`/`max` needed the `as_numeric` workaround to stay
   `Numeric`-homed. **Resolved: deferred dispatch.** Forward edges now record a
   `BackwardSource` (operands + a `BackwardOpKind` marker) with *no* `GradRule`
   bound; `BackwardPlan::build` materializes the concrete rule
   (`BackwardSource::into_grad_fn`, `T: Float`) once per node and `run` executes
   it. Forward homes are now chosen purely by kernels; rules only by their own
   math; `sub`/`div` are forward for every `Numeric` (forward-only for unsigned
   ints), `neg` remains `Signed`.

## Divergences — resolved, but with still-open consequences

All five divergences are resolved (see above). What remains open is tracked as
one-item-per-session tickets (each is self-contained and independently
verifiable):

- [ ] **Session: Signed-family fallout** — `Signed` family landed
  (`dtype/mod.rs`): `neg` stays `Signed`; forward `sub`/`div` widened to
  `Numeric` (unsigned ints get forward-only ops) via deferred dispatch
  (divergences 2/5). `as_numeric` (`tensor/cast.rs`) survives as a `Numeric`-
  pure mask reinterpretation for `maximum`/`max` rules — only ever dispatched
  at backward time, so it is no longer a forward-signature workaround.
- [ ] **Session: differentiable casts** — `tensor/cast.rs` casts are currently
  a "non-differentiable data transform" (no gradient edge). Adding edges would
  let a graph flow through a dtype boundary.
- [ ] **Session: graph-level `abs`/`modul`** — only exist at the storage level;
  wire them onto tensors (an `abs` bounded by `T: Signed` and a
  non-differentiable `modul` on `T: Numeric`).
- [ ] **Session: `f16`/`bf16` future members** — `dtype/mod.rs` documents them
  as future `Float` members; would need cast edges into the existing table.
- [ ] **Session: `cargo fmt` sweep** — `cargo fmt --check` reports pre-existing
  formatting debt across the Stage 0–4 files (and the merge test files); the
  `make check` gate includes `cargo fmt --check`.
- [ ] **Session: `clippy --all-targets` debt** — `cargo clippy --all-targets
  -- -D warnings` fails on pre-existing lints in the legacy test files
  (loop-variable indexing, literal-bool asserts, `vec!`, unnecessary parens).
  `make check` uses lib-only clippy and is green.
- [ ] **Session: commit the work** — all stages + the deferred-dispatch refactor
  are uncommitted on `main` (`git status` shows the whole tree).

See also `SESSION_CHANGES.md` (older roadmap: SIMD/threaded matmul, dead-code
cleanup).