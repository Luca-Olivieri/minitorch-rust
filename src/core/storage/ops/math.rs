use crate::core::dtype::{Float, Numeric, Signed};
use crate::core::storage::TensorStorage;

impl<T: Numeric> TensorStorage<T> {
    // Not every storage-level op is reachable from the user-facing tensor layer yet,
    // so unused ones are explicitly allowed here.
    //
    // add/sub/mul/div/modul/maximum are signed-agnostic, so they live on the full
    // `Numeric` family: integer tensors get them too (with integer overflow semantics).
    impl_storage_elemwise_ops!(TensorStorage<T>;
        add,     (a, b), a + b;
        sub,     (a, b), a - b;
        mul,    (a, b), a * b;
        div,     (a, b), a / b;
        modul,   (a, b), a % b;
        maximum, (a, b), if a > b { a } else { b };
    );

    /// out[i] = a[i] - scale * b[i], fused into a single pass.
    pub fn sub_scaled(a: &TensorStorage<T>, b: &TensorStorage<T>, scale: T) -> TensorStorage<T> {
        crate::core::storage::ops::utils::apply_op(&[a, b], |&[av, bv]| av - scale * bv)
    }

    /// Direct [m,k] x [k,n] -> [m,n] GEMM kernel.
    ///
    /// Three paths, selected on the runtime strides of `b`:
    ///
    /// * `b_s1 == 1` — output-row tiling. A fixed-size stack tile holds the
    ///   partial sums for [`GEMM_TILE`] output columns across the whole `k`
    ///   reduction, so they stay in vector registers instead of being reloaded
    ///   from `out_buf` once per `k` step. Also lets the compiler see a
    ///   contiguous inner access and vectorize it.
    /// * `b_s0 == 1` — dot-product form. Swaps the loops so `k` is innermost.
    ///   This is the layout `dL/dA = grad @ W^T` produces, where `W` is stored
    ///   row-major so the transpose leaves the *outer* axis unit-stride. Reading
    ///   `k` innermost then streams both operands contiguously instead of
    ///   touching every output column at a full row stride.
    /// * neither — the original generic loop.
    ///
    /// All three accumulate each output element over `k = 0, 1, ..., k-1` in the
    /// same order, so results are bit-identical to the generic loop.
    pub fn matmul(a: &TensorStorage<T>, b: &TensorStorage<T>) -> TensorStorage<T> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            panic!(
                "TensorStorage::matmul requires 2D operands, got {:?} and {:?}.",
                a.shape, b.shape
            );
        }

        if a.shape[1] != b.shape[0] {
            panic!(
                "matmul inner dimensions must match ({} != {}).",
                a.shape[1], b.shape[0]
            );
        }

        // `out_buf` is freshly allocated and uniquely owned, so it is mutable.
        let mut out_buf = vec![T::ZERO; a.shape[0] * b.shape[1]];

        let layout = GemmLayout {
            a_off: a.offset,
            b_off: b.offset,
            a_s0: a.strides[0],
            a_s1: a.strides[1],
            b_s0: b.strides[0],
            b_s1: b.strides[1],
            m: a.shape[0],
            k: a.shape[1],
            n: b.shape[1],
        };

        if layout.b_s1 == 1 {
            Self::gemm_row_tiled(&mut out_buf, &a.buffer, &b.buffer, &layout);
        } else if layout.b_s0 == 1 {
            Self::gemm_dot_product(&mut out_buf, &a.buffer, &b.buffer, &layout);
        } else {
            for i in 0..layout.m {
                let a_row_base = layout.a_off + i * layout.a_s0;
                let out_row_base = i * layout.n;
                for kk in 0..layout.k {
                    let a_val = a.buffer[a_row_base + kk * layout.a_s1];
                    let b_row_base = layout.b_off + kk * layout.b_s0;
                    for j in 0..layout.n {
                        out_buf[out_row_base + j] += a_val * b.buffer[b_row_base + j * layout.b_s1];
                    }
                }
            }
        }

        TensorStorage::from_buffer(vec![a.shape[0], b.shape[1]], out_buf)
    }

    /// Output-row-tiled GEMM for a column-contiguous `b` (`b_s1 == 1`).
    ///
    /// The indexed `tile` access is deliberate: a fixed-size local array is
    /// what the optimizer needs in order to scalar-replace the partial sums
    /// into vector registers across the `k` loop. An iterator or a `Vec` would
    /// hide the trip count and leave the accumulator in memory.
    #[allow(clippy::needless_range_loop)]
    fn gemm_row_tiled(out_buf: &mut [T], a_buf: &[T], b_buf: &[T], g: &GemmLayout) {
        let tiled = g.n - g.n % GEMM_TILE;

        for i in 0..g.m {
            let a_row_base = g.a_off + i * g.a_s0;
            let out_row_base = i * g.n;

            for j in (0..tiled).step_by(GEMM_TILE) {
                let mut tile = [T::ZERO; GEMM_TILE];
                for kk in 0..g.k {
                    let a_val = a_buf[a_row_base + kk * g.a_s1];
                    // One range check per `k` step instead of one per tile
                    // element: the sub-slice length is a constant, so the
                    // indexed loads below need no further bounds checks.
                    let b_row = &b_buf[g.b_off + kk * g.b_s0 + j..][..GEMM_TILE];
                    for u in 0..GEMM_TILE {
                        tile[u] += a_val * b_row[u];
                    }
                }
                out_buf[out_row_base + j..out_row_base + j + GEMM_TILE].copy_from_slice(&tile);
            }

            // Columns left over when `n` is not a multiple of the tile width.
            for j in tiled..g.n {
                let mut sum = T::ZERO;
                for kk in 0..g.k {
                    sum += a_buf[a_row_base + kk * g.a_s1] * b_buf[g.b_off + kk * g.b_s0 + j];
                }
                out_buf[out_row_base + j] = sum;
            }
        }
    }

    /// Dot-product GEMM for a row-contiguous `b` (`b_s0 == 1`).
    ///
    /// Each output element is a single independent dot product, so the output
    /// row is written exactly once and `b`'s unit-stride axis becomes the
    /// innermost access.
    fn gemm_dot_product(out_buf: &mut [T], a_buf: &[T], b_buf: &[T], g: &GemmLayout) {
        for i in 0..g.m {
            let a_row_base = g.a_off + i * g.a_s0;
            let out_row_base = i * g.n;
            for j in 0..g.n {
                let mut sum = T::ZERO;
                let b_col_base = g.b_off + j * g.b_s1;
                for kk in 0..g.k {
                    sum += a_buf[a_row_base + kk * g.a_s1] * b_buf[b_col_base + kk];
                }
                out_buf[out_row_base + j] = sum;
            }
        }
    }
}

/// Extents and element strides shared by every [`TensorStorage::matmul`] path.
struct GemmLayout {
    a_off: usize,
    b_off: usize,
    a_s0: usize,
    a_s1: usize,
    b_s0: usize,
    b_s1: usize,
    /// Rows of `a`, the output's row count.
    m: usize,
    /// Shared inner dimension.
    k: usize,
    /// Columns of `b`, the output's column count.
    n: usize,
}

/// Output elements held in registers at once by [`TensorStorage::gemm_row_tiled`].
///
/// Eight `f64` occupy four 128-bit NEON registers, matching the accumulator
/// width the convolution kernels already reach on this target.
const GEMM_TILE: usize = 8;

impl<T: Signed> TensorStorage<T> {
    // `neg` and `abs` need a signed notion: `neg` is undefined for unsigned
    // integers (`std` has no `Neg` for `u8`…`u64`) and `abs` for them is the
    // identity, so both live on `Signed` (floats and signed integers).
    impl_storage_elemwise_ops!(TensorStorage<T>;
        neg,     (a), -a;
        abs,     (a), a.abs();
    );
}

impl<T: Float> TensorStorage<T> {
    // pow/ln/exp/sqrt are transcendental, so float-only.
    impl_storage_elemwise_ops!(TensorStorage<T>;
        pow,     (b, e), b.powf(e);
        ln,     (a), a.ln();
        exp,     (a), a.exp();
        sqrt,     (a), a.sqrt();
    );
}
