//! Dtype system: the closed set of element types a tensor may carry, the trait
//! families used to scope operations, and the compile-time conversion tiers.
//!
//! # Families
//!
//! - [`Dtype`] — base marker; implemented for every supported scalar type.
//! - [`Numeric`] — arithmetic-capable types (`+ - * /`): [`Float`] and
//!   [`Integer`].
//! - [`Float`] — IEEE 754 floats (`f32`, `f64`).
//! - [`Integer`] — signed and unsigned integers (`i8`…`i64`, `u8`…`u64`).
//!
//! `bool` implements only [`Dtype`]: it is not arithmetic, so any operation
//! gated on [`Numeric`] does not exist for boolean tensors. Logical operations
//! target `Tensor<bool>` directly.
//!
//! # Conversion model
//!
//! An element conversion is a directed edge declared as a trait impl, so a
//! conversion is legal exactly when the impl exists: compilation decides
//! legality, there is no runtime dispatch and no silent coercion. Edges are
//! split into three tiers with distinct failure semantics:
//!
//! - [`CastFrom`] — exact and infallible (widening only). Always compiled.
//! - [`LossyCastFrom`] — total but rounding (`f64 -> f32`, `i64 -> f64`).
//!   Compiled only with the `allow_lossy_casts` cargo feature.
//! - [`DangerousCastFrom`] — partial; panics when the input is outside the
//!   target domain (e.g. `(u32) 5.5`). Compiled only with the
//!   `allow_dangerous_casts` cargo feature.
//!
//! The two `allow_*` features are independent: enabling one never enables the
//! other, and with both disabled the lossy and dangerous methods do not exist
//! in the compiled crate at all.

mod casts;

mod sealed {
    pub trait Sealed {}
}

pub use casts::CastFrom;
#[cfg(feature = "allow_lossy_casts")]
pub use casts::LossyCastFrom;
#[cfg(feature = "allow_dangerous_casts")]
pub use casts::DangerousCastFrom;

use std::fmt;

/// Base family: every scalar type a tensor may carry.
///
/// Sealed to the primitives implemented below, so the dtype universe is a
/// closed set known at compile time.
#[allow(private_bounds)]
pub trait Dtype: sealed::Sealed + Copy + Clone + PartialEq + fmt::Debug + Send + Sync + 'static {}

/// Arithmetic family (`+`, `-`, `*`, `/`, `%`), the common ground of [`Float`]
/// and [`Integer`]. Ops that make sense for every number (e.g. `matmul`) are
/// gated on this; `bool` is excluded by construction.
///
/// `ZERO`/`ONE` are the only distinguished scalars an op may need (e.g. to
/// seed a sum or build a 1/0 mask without a promotion to `f64`).
#[allow(private_bounds)]
pub trait Numeric:
    Dtype
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Rem<Output = Self>
    + std::ops::AddAssign
    + std::cmp::PartialOrd
{
    const ZERO: Self;
    const ONE: Self;
}

/// IEEE 754 floating-point family.
///
/// Future members (not currently required, and not std types today):
/// - half precision `f16`, bfloat16 `bf16` — would need custom scalar types.
#[allow(private_bounds)]
pub trait Float: Numeric + std::ops::Neg<Output = Self> {
    /// Reinterpret a float as an `f64` (exact: floats only ever widen).
    #[doc(hidden)]
    fn to_f64(self) -> f64;
    /// Reinterpret an `f64` as this float type (rounds for `f32`).
    #[doc(hidden)]
    fn from_f64(x: f64) -> Self;
    #[doc(hidden)]
    fn powf(self, e: Self) -> Self;
    #[doc(hidden)]
    fn ln(self) -> Self;
    #[doc(hidden)]
    fn exp(self) -> Self;
    #[doc(hidden)]
    fn abs(self) -> Self;
    #[doc(hidden)]
    fn sqrt(self) -> Self;
}

/// Signed and unsigned integer family.
///
/// Future split (not currently required):
/// - `SignedInteger: Integer` / `UnsignedInteger: Integer` — the ops that
///   genuinely need signedness are `neg`/`abs` (undefined for unsigned) and
///   the differentiable `sub`/`div` (whose backward math negates the gradient).
///   They are therefore currently gated on [`Float`], and would be widened to
///   a `SignedInteger` family if one is added.
#[allow(private_bounds)]
pub trait Integer: Numeric {}

impl sealed::Sealed for bool {}
impl Dtype for bool {}

macro_rules! impl_float_family {
    ($($t:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $t {}
            impl Dtype for $t {}
            impl Numeric for $t {
                const ZERO: Self = 0.0;
                const ONE: Self = 1.0;
            }
            impl Float for $t {
                fn to_f64(self) -> f64 { self as f64 }
                fn from_f64(x: f64) -> Self { x as $t }
                fn powf(self, e: Self) -> Self { <$t>::powf(self, e) }
                fn ln(self) -> Self { self.ln() }
                fn exp(self) -> Self { self.exp() }
                fn abs(self) -> Self { self.abs() }
                fn sqrt(self) -> Self { self.sqrt() }
            }
        )+
    };
}

macro_rules! impl_integer_family {
    ($($t:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $t {}
            impl Dtype for $t {}
            impl Numeric for $t {
                const ZERO: Self = 0;
                const ONE: Self = 1;
            }
            impl Integer for $t {}
        )+
    };
}

impl_float_family!(f32, f64);
impl_integer_family!(i8, i16, i32, i64, u8, u16, u32, u64);

/// Internal per-element rendering used by tensor `Display` impls.
///
/// Floats render with a fixed precision (mirroring the pre-generic `{:.4}`
/// output); integers and `bool` render plainly.
///
/// This is an implementation detail of tensor formatting; it is public only so
/// the public [`super::tensor::AbstractTensor`]-related impls can name it in
/// their bounds.
#[doc(hidden)]
pub trait DtypeStyler: Dtype {
    fn dtype_name() -> &'static str;
    fn fmt_val(val: &Self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

macro_rules! impl_styler_float {
    ($($t:ty => $name:literal),+ $(,)?) => {
        $(
            impl DtypeStyler for $t {
                fn dtype_name() -> &'static str { $name }
                fn fmt_val(val: &Self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:.4}", *val)
                }
            }
        )+
    };
}

macro_rules! impl_styler_plain {
    ($($t:ty => $name:literal),+ $(,)?) => {
        $(
            impl DtypeStyler for $t {
                fn dtype_name() -> &'static str { $name }
                fn fmt_val(val: &Self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", *val)
                }
            }
        )+
    };
}

impl_styler_float!(f32 => "float32", f64 => "float64");
impl_styler_plain!(
    i8 => "int8",
    i16 => "int16",
    i32 => "int32",
    i64 => "int64",
    u8 => "uint8",
    u16 => "uint16",
    u32 => "uint32",
    u64 => "uint64",
    bool => "bool",
);