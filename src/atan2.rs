//! Fast atan2 approximations.
// Taken from: https://mazzo.li/posts/vectorized-atan2.html
// The original license header is reproduced below.
//
// Copyright (c) 2021 Francesco Mazzoli <f@mazzo.li>
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

use core::f32;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m256;

const A1: f32 = 0.99997726;
const A3: f32 = -0.33262347;
const A5: f32 = 0.19354346;
const A7: f32 = -0.11643287;
const A9: f32 = 0.05265332;
const A11: f32 = -0.0117212;

/// Fast atan2 approximation.
///
/// # Panics
///
/// - if start of `xs` and `ys` is not aligned to 32-bytes
/// - if `xs` and `ys` are not the same length
/// - if the input includes the point (0., 0.).
pub fn atan2_inplace(xs_out: &mut [f32], ys: &[f32]) {
    assert_eq!(xs_out.len(), ys.len());
    assert_eq!(ys.as_ptr() as usize % 32, 0);

    xs_out
        .iter_mut()
        .zip(ys.iter().copied())
        .for_each(|(x, y)| *x = atan2_single(*x, y));
}

#[inline(always)]
pub fn atan2_single(x: f32, y: f32) -> f32 {
    assert!(x != 0. || y != 0.);
    let swap = x.abs() < y.abs();
    let a = if swap { x / y } else { y / x };
    let asq = a * a;
    let atan_res = a * (A1 + asq * (A3 + asq * (A5 + asq * (A7 + asq * (A9 + asq * A11)))));
    let res = if swap {
        f32::consts::FRAC_PI_2.copysign(a) - atan_res
    } else {
        atan_res
    };
    if x.is_sign_negative() {
        f32::consts::PI.copysign(y) + res
    } else {
        res
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline]
pub unsafe fn atan2_avx2(x: __m256, y: __m256) -> __m256 {
    use std::arch::x86_64::*;
    let a1 = _mm256_set1_ps(A1);
    let a3 = _mm256_set1_ps(A3);
    let a5 = _mm256_set1_ps(A5);
    let a7 = _mm256_set1_ps(A7);
    let a9 = _mm256_set1_ps(A9);
    let a11 = _mm256_set1_ps(A11);
    let pi = _mm256_set1_ps(f32::consts::PI);
    let pi_2 = _mm256_set1_ps(f32::consts::FRAC_PI_2);

    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
    let abs_y = _mm256_and_ps(abs_mask, y);
    let abs_x = _mm256_and_ps(abs_mask, x);
    let a = _mm256_div_ps(_mm256_min_ps(abs_x, abs_y), _mm256_max_ps(abs_x, abs_y));

    let asq = _mm256_mul_ps(a, a);

    let result = a11;
    let result = _mm256_fmadd_ps(asq, result, a9);
    let result = _mm256_fmadd_ps(asq, result, a7);
    let result = _mm256_fmadd_ps(asq, result, a5);
    let result = _mm256_fmadd_ps(asq, result, a3);
    let result = _mm256_fmadd_ps(asq, result, a1);
    let result = _mm256_mul_ps(a, result);

    let swap_mask = _mm256_cmp_ps(abs_y, abs_x, _CMP_GT_OQ);
    let result = _mm256_add_ps(
        _mm256_xor_ps(result, _mm256_and_ps(sign_mask, swap_mask)),
        _mm256_and_ps(pi_2, swap_mask),
    );

    let x_sign_mask = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(x), 31));
    let result = _mm256_add_ps(
        _mm256_xor_ps(result, _mm256_and_ps(x, sign_mask)),
        _mm256_and_ps(pi, x_sign_mask),
    );

    _mm256_xor_ps(result, _mm256_and_ps(y, sign_mask))
}
