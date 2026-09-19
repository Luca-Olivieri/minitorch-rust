//! Tests for the dtype families and the compile-time conversion tiers.
//!
//! The exact tier must always compile; the lossy and dangerous tiers are gated
//! behind their cargo features, so compile them only when enabled. Absence of
//! an edge is itself the guarantee (a missing impl is a compile error), which
//! cannot be tested at runtime without `trybuild`.

use crate::core::dtype::CastFrom;

#[test]
fn identity_is_available() {
    assert_eq!(f64::cast_from(3.5_f64), 3.5);
    assert!(bool::cast_from(true));
    assert_eq!(i32::cast_from(7_i32), 7);
}

#[test]
fn bool_embeds_exactly_everywhere() {
    assert_eq!(i32::cast_from(false), 0);
    assert_eq!(i8::cast_from(true), 1);
    assert_eq!(f64::cast_from(true), 1.0);
    assert_eq!(f32::cast_from(false), 0.0);
}

#[test]
fn integer_widening_is_bit_perfect() {
    assert_eq!(i64::cast_from(i32::MIN), i32::MIN as i64);
    assert_eq!(u64::cast_from(u32::MAX), u32::MAX as u64);
    assert_eq!(i32::cast_from(i16::MIN), i16::MIN as i32);
    assert_eq!(i64::cast_from(u32::MAX), u32::MAX as i64);
}

#[test]
fn int_and_unsigned_to_f64_is_exact() {
    assert_eq!(f64::cast_from(i32::MAX), 2_147_483_647.0);
    assert_eq!(f64::cast_from(u32::MAX), 4_294_967_295.0);
}

#[test]
fn f32_to_f64_roundtrips_exactly() {
    let x = 0.1_f32;
    let wide = f64::cast_from(x);
    assert_eq!(wide as f32, x);
}

#[cfg(feature = "allow_lossy_casts")]
mod lossy {
    use crate::core::dtype::LossyCastFrom;

    #[test]
    fn f64_to_f32_rounds() {
        assert_eq!(f32::lossy_cast_from(0.1_f64), 0.1_f64 as f32);
    }

    #[test]
    fn f64_to_f32_overflows_to_inf() {
        assert_eq!(f32::lossy_cast_from(1e300_f64), f32::INFINITY);
        assert_eq!(f32::lossy_cast_from(-1e300_f64), f32::NEG_INFINITY);
    }

    #[test]
    fn i32_to_f32_rounds() {
        // 16_777_217 needs 25 bits; the f32 mantissa keeps 24.
        assert_eq!(f32::lossy_cast_from(16_777_217_i32), 16_777_216.0);
    }

    #[test]
    fn i64_to_f64_rounds() {
        assert_eq!(f64::lossy_cast_from(i64::MAX), 9_223_372_036_854_775_808.0);
    }

    #[test]
    fn u64_to_f64_rounds() {
        assert_eq!(f64::lossy_cast_from(u64::MAX), 18_446_744_073_709_551_616.0);
    }

    #[test]
    fn numeric_to_bool_is_total() {
        assert!(bool::lossy_cast_from(1_i32));
        assert!(!bool::lossy_cast_from(0_i8));
        assert!(bool::lossy_cast_from(-3.5_f64));
        assert!(!bool::lossy_cast_from(-0.0_f64));
    }
}

#[cfg(feature = "allow_dangerous_casts")]
mod dangerous {
    use crate::core::dtype::DangerousCastFrom;

    #[test]
    fn float_to_int_succeeds_when_integral() {
        assert_eq!(i32::dangerous_cast_from(-7.0_f64), -7);
        assert_eq!(u8::dangerous_cast_from(255.0_f64), 255);
        assert_eq!(i64::dangerous_cast_from(2.0_f32), 2);
        assert_eq!(i32::dangerous_cast_from(-0.0_f64), 0);
    }

    #[test]
    #[should_panic(expected = "integral")]
    fn float_to_int_panics_on_fraction() {
        let _ = i32::dangerous_cast_from(5.5_f64);
    }

    #[test]
    #[should_panic]
    fn float_to_unsigned_panics_on_negative() {
        let _ = u32::dangerous_cast_from(-1.0_f64);
    }

    #[test]
    #[should_panic]
    fn float_to_int_panics_on_out_of_range() {
        let _ = u8::dangerous_cast_from(256.0_f64);
    }

    #[test]
    #[should_panic]
    fn float_to_u64_rejects_2_64() {
        let _ = u64::dangerous_cast_from(18_446_744_073_709_551_616.0_f64);
    }

    #[test]
    #[should_panic]
    fn float_to_int_panics_on_nan_and_inf() {
        let _ = i32::dangerous_cast_from(f64::NAN);
    }

    #[test]
    #[should_panic]
    fn u64_rejects_infinite() {
        let _ = u64::dangerous_cast_from(f64::INFINITY);
    }

    #[test]
    fn signed_to_unsigned_ok_when_non_negative() {
        assert_eq!(u32::dangerous_cast_from(42_i64), 42);
        assert_eq!(u64::dangerous_cast_from(7_i64), 7);
    }

    #[test]
    #[should_panic]
    fn signed_to_unsigned_panics_on_negative() {
        let _ = u8::dangerous_cast_from(-1_i8);
    }

    #[test]
    fn narrowing_succeeds_in_range() {
        assert_eq!(i16::dangerous_cast_from(300_i32), 300);
        assert_eq!(u8::dangerous_cast_from(255_u64), 255);
    }

    #[test]
    #[should_panic]
    fn narrowing_panics_out_of_range() {
        let _ = i8::dangerous_cast_from(200_i16);
    }

    #[test]
    #[should_panic]
    fn unsigned_to_signed_panics_out_of_range() {
        let _ = i32::dangerous_cast_from(4_000_000_000_u64);
    }

    #[test]
    #[should_panic]
    fn f32_to_i64_panics_out_of_range() {
        let _ = i64::dangerous_cast_from(1e30_f32);
    }
}
