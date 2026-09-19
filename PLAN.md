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

## Stage 6 — Differentiable casts + edge hygiene (done)

Two locked rules drive this stage:

1. **No edge for non-differentiable work.** A forward node records a graph edge
   (`grad_fn = Some(BackwardSource)`) *iff* the operation is differentiable for
   its operand dtype. Integer/bool arithmetic, comparisons, logical ops,
   `argmax`, `one_hot`, `stack`, `sub_scaled` and non-differentiable casts are
   graph boundaries (`grad_fn = None`). Before 6a, integer `add`/`mul`/`sub`/
   `div`/`neg` and every `Numeric`/`Signed` reduction still attached an edge that
   could never be backpropagated (backward runs only for `T: Float`); 6a removed
   them.
2. **A cast is differentiable iff both dtypes are `Float` *and* the reverse
   conversion is legal in the compiled crate.** Concretely:
   - `f32 -> f64` (exact) gets an edge only with `allow_lossy_casts`, because its
     only reverse is the lossy `f64 -> f32` (hand-written float block in
     `casts.rs`). Without the feature it stays forward-only: it compiles, but
     carries no edge and cannot be backpropagated through.
   - `f64 -> f32` (lossy, only exists with the feature) gets an edge whenever it
     exists, since the reverse `f32 -> f64` is exact.
   - int↔float / int↔int / bool↔anything casts: no edge (an integer side never
     backprops).
   A differentiable cast's backward applies the reverse conversion to the
   upstream gradient, yielding a gradient in the *source* dtype (the
   "gradient w.r.t. an input has that input's dtype" invariant).

### 6a — Edge hygiene (done)

- Implemented witness: `const DIFFERENTIABLE: bool` on `Dtype` (`true` only for
  `f32`/`f64`), consumed by a single gate `maybe_edge` in `grad_fn.rs`. Chosen
  over the planned associated marker type because a const needs no
  private-type bound on the public `Dtype` trait and no witness impls; it is
  still fully compile-time (monomorphized constant, dead branch eliminated).
- All edge sites now route through the gate: `apply_tensor_op` takes an
  `Option<BackwardOpKind>` (not a builder closure), and `copy_d`, `matmul`,
  `sum`, `max` and the shape ops call `maybe_edge` directly. Non-differentiable
  sites (comparisons, logic, `argmax`, `one_hot`, `stack`, `sub_scaled`) stay
  edge-free.
- **Resolved — `requires_grad` propagation:** an edge is now required for
  propagation (`requires_grad = edge.is_some() && any(operand)`), so
  non-differentiable outputs report `false`.
- Tests updated: `int_sub_neg_carry_graph_edges` → `int_ops_are_graph_boundaries`
  and the last block of `unsigned_sub_div` now assert boundaries/false.
- Verified: default lib 40 pass/2 ignored, models 9, nn 1, tensor 58;
  all-features lib 67/2, models 9, nn 1, tensor 58; `clippy -D warnings` clean;
  `cargo build --bins` OK; xor parity (dist 0.3430100583, `grads_map.len()=6`).
- No forward value and no float backward path changed: the delta is graph
  structure for non-float dtypes only.

### 6b — Scheduler erasure (done)

- New `autograd/erased.rs` holds the erased vocabulary:
  - `GradValue { F32(GraphTensor<f32>), F64(GraphTensor<f64>) }` — the plan's
    gradient slots.
  - `ErasedTensor` — a closed two-variant node handle that is also the plan's
    key (identity = `Rc` pointer) and exposes `requires_grad`/`is_leaf`/`shape`/
    `erased_operands`/`one_grad`.
  - `ErasedHandle(Rc<dyn Any>)` — the untyped handle an edge stores. Forward
    edges are built in generic code, so erasure cannot name the dtype at
    construction time; `maybe_edge` erases for *every* `T: Dtype` (the cast is
    valid because every dtype is `'static`) and `BackwardPlan::build` checks it
    into `ErasedTensor`.
  - `FloatRepr` — crate-internal bridge between a concrete float and the erased
    enums. The erasure is total because `Float` is sealed to `f32`/`f64`; the
    panic arms are unreachable by construction.
- `BackwardSource` loses its `T`: it is `{ Vec<ErasedHandle>, BackwardOpKind }`,
  so `TensorNode<T>::grad_fn` is `Option<Box<BackwardSource>>` and an edge can
  connect different float dtypes (the prerequisite for 6c).
- Materialization: `materialize_grad_fn` is the single closed `f32`/`f64` match.
  It boxes the typed rule behind an object-safe `ErasedGradFn`, whose
  `TypedGradFn<T>` carries the rule and its reusable scratch buffer and converts
  typed results into `GradValue` slots (no per-node allocation after warmup).
  The typed rule math (`GradRule`/`NBackwardOp`) is unchanged.
- `BackwardPlan` is now non-generic; `run<T: FloatRepr>` keeps the *public*
  `backward`/`compile_backward`/optimizer signatures typed (`HashMap<TensorKey<T>,
  GraphTensor<T>>`) by projecting the erased map (downcast). **Deviation from
  the plan text:** the public heterogeneous API and its caller ripple (tests,
  `xor`, `covertype`) move to 6c, the first stage where one backward pass can
  actually produce two gradient dtypes. In 6b every requires-grad graph is still
  monomorphic, so the typed projection is exact and no caller changes.
- No behavior change: default lib 40/2, models 9, nn 1, tensor 58; all-features
  lib 67/2, models 9, nn 1, tensor 58; `clippy -D warnings` clean (default and
  all-features); `cargo build --bins` OK; xor parity (dist `0.3430100583`, 6
  grads). Higher-order/`retain_graph` (`tests/tensor/math.rs`) and `f32_backward`
  (`tests/tensor/dtypes.rs`) pass, exercising both erased arms.

### 6c — Differentiable casts + erased public `backward` API (done)

- `CastBackward<Src: Dtype>: Dtype` (`dtype/casts.rs`) is the per-pair companion
  witness: `const GRAD_EDGE: bool = false` for every pair the cast-table macros
  generate, with real impls for the float pairs:
  - `f32 -> f32`, `f64 -> f64` (identity) and `f64 -> f32` (lossy narrowing) —
    edge always (the reverse is exact or, for identity, itself).
  - `f32 -> f64` (exact widening) — edge iff `allow_lossy_casts`, because the
    reverse `f64 -> f32` backcast is lossy (`GRAD_EDGE = cfg!(...)`).
  The float↔float pairs moved out of the table macros into a hand-written block:
  their right to an edge depends on the *reverse* tier, which a single
  forward-tier table cannot express.
- `GraphTensor::cast`/`cast_lossy`/`cast_dangerous` now bound `U: CastBackward<T>`
  and attach `BackwardOpKind::CastOp` iff `U::GRAD_EDGE`;
  `requires_grad = edge.is_some() && operand.requires_grad`, so every
  non-differentiable cast is a graph boundary (int/float, bool, dangerous).
- `BackwardOpKind::CastOp` is a unit variant: the destination dtype is the node's
  own dtype and the source is read from the single erased operand at
  materialization time. `ErasedCastGradFn` does the closed `(src, dst)` match and
  applies the *reverse* cast, reusing the forward `.cast::<_>()`/`.cast_lossy::<_>()`
  so the backcast is itself a graph node (higher-order through a cast works).
- Public API: `backward` now returns the erased `GradMap` (no additive
  `backward_erased`). A typed `HashMap<TensorKey<T>, GraphTensor<T>>` cannot key
  an `f32` leaf in an `f64` map, and a single backward pass over a mixed graph
  produces both. `GradMap` is keyed by node identity and reads back through
  `get::<T>(&tensor) -> Option<&GraphTensor<T>>` (non-panicking; `None` also
  covers asking for the wrong float). `TensorKey`, `to_key`,
  `ErasedTensor::into_typed` and the typed `FloatRepr` projection helpers were
  deleted. Callers updated mechanically (`grads.get(&t)`,
  `grads.get(&t).is_none()`); `Optimizer::step` now takes `&GradMap`.
- Tests (`tests/tensor/dtypes.rs`): identity cast is differentiable; `f64 -> f32`
  backcast reaches the `f64` leaf; `f32 -> f64` reaches the `f32` leaf under
  `allow_lossy_casts`; a mixed graph keeps one gradient per leaf dtype
  (`grads.len() == 2`); `int -> float` stays a boundary; widening is
  forward-only without the feature (`!y.requires_grad()`); a cast is twice
  differentiable.
- Verification: default lib 40/2, models 9, nn 1, tensor 60; all-features lib
  67/2, models 9, nn 1, tensor 64; `clippy -D warnings` clean (default and
  all-features); `cargo build --bins` OK; xor parity unchanged (dist
  `0.3430100583`, 6 grads).

### 6d — Docs (done)

- Divergence #6 recorded above; `CastBackward`, `GraphTensor::cast*`, the erased
  autograd scheduler and the `GradMap` all carry their own module/item docs; the
  `PLAN.md` tickets below are updated. The old "casts are a non-differentiable
  data transform" note is gone (see 6c).

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
6. **Stage 6 cast rule shape and public `backward` result.** The plan described a
   companion trait "generated by the cast-table macros" that both states whether a
   pair has a legal reverse *and* performs the backcast, and a public `backward`
   returning "a `GradValue`-keyed map or a typed accessor". Chosen instead:
   the macros emit marker impls (`GRAD_EDGE = false`) for every pair and only the
   float↔float pairs are hand-written, because whether `f32 -> f64` has an edge
   depends on the *reverse* tier (`allow_lossy_casts`), which a single forward-tier
   table cannot express; the backcast is not duplicated in the trait but applied by
   `ErasedCastGradFn`, which reuses the forward cast op so the backcast is itself a
   graph node (higher-order through a cast works). `backward` returns the erased
   `GradMap` directly — no additive typed accessor — because a typed key cannot
   represent a graph holding both an `f32` and an `f64` leaf.

## Divergences — resolved, but with still-open consequences

All six divergences are resolved (see above). What remains open is tracked as
one-item-per-session tickets (each is self-contained and independently
verifiable):

- [x] **Session: Signed-family fallout** — `Signed` landed (`dtype/mod.rs`):
  `neg` and `abs` stay `Signed`; forward `sub`/`div` widened to `Numeric`
  (unsigned ints get forward-only ops) via deferred dispatch (divergences 2/5);
  `Signed::abs` is a required trait method (`wrapping_abs` for signed ints).
  `abs` is wired onto tensors at the `Signed` home (differentiable for floats,
  boundary for ints) and `modul` at the `Numeric` home (non-differentiable
  boundary). `as_numeric` (`tensor/cast.rs`) survives as the `Numeric`-pure mask
  reinterpretation feeding the `maximum`/`max`/`abs` rules — only ever dispatched
  at backward time, so it is no longer a forward-signature workaround.
- [x] **Session: edge hygiene (Stage 6a)** — done: `Dtype::DIFFERENTIABLE` +
  `maybe_edge` gate; non-differentiable ops are graph boundaries and do not
  propagate `requires_grad`. See the Stage 6a notes above.
- [x] **Session: differentiable casts (Stage 6c–6d)** — done: `CastBackward`
  witness + reverse-legality-gated float cast edges, erased `CastOp` rule, the
  public erased `GradMap` returned by `backward` (replaces the typed projection),
  and divergence #6 docs. See the Stage 6c/6d notes above. Replaces the old
  "casts are a non-differentiable data transform" note.
- [x] **Session: `requires_grad` semantics for non-differentiable ops** —
  resolved in 6a: propagation now requires an edge (`false` for non-diff
  outputs).
- [x] **Session: graph-level `abs`/`modul`** — wired in the signed-family session
  above: `GraphTensor<T>::abs` on `T: Signed` (float-`sign` gradient via
  `AbsOp`; boundary for ints) and `GraphTensor<T>::modul` on `T: Numeric`
  (non-differentiable boundary). Tested in `tests/tensor/math.rs`.
- [ ] **Session: `f16`/`bf16` future members** — `dtype/mod.rs` documents them
  as future `Float` members; would need cast edges into the existing table.
- [ ] **Session: `cargo fmt` sweep** — `cargo fmt --check` reports pre-existing
  formatting debt across the Stage 0–4 files (and the merge test files); the
  `make check` gate includes `cargo fmt --check`.
- [x] **Session: `clippy --all-targets` debt** — `cargo clippy --all-targets
  -- -D warnings` is now clean: the loop-variable indexing, literal-bool
  asserts, `vec!`-of-arrays, same-type casts and unnecessary parens in the
  unit/tensor/nn/models suites were all tidied. `make check` stays green.
- [ ] **Session: commit the work** — all stages + the deferred-dispatch refactor
  are uncommitted on `main` (`git status` shows the whole tree).

See also `SESSION_CHANGES.md` (older roadmap: SIMD/threaded matmul, dead-code
cleanup).