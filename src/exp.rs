/// Fast exponential approximation.
// Based on: https://stackoverflow.com/a/48869291 by SO user wim.
// Originally based on https://github.com/reyoung/avx_mathfun.
// The original license header is reproduced below.
//
//   AVX implementation of exp
//   Based on "sse_mathfun.h", by Julien Pommier
//   http://gruntthepeon.free.fr/ssemath/
//   Copyright (C) 2012 Giovanni Garberoglio
//   Interdisciplinary Laboratory for Computational Science (LISC)
//   Fondazione Bruno Kessler and University of Trento
//   via Sommarive, 18
//   I-38123 Trento (Italy)
//  This software is provided 'as-is', without any express or implied
//  warranty.  In no event will the authors be held liable for any damages
//  arising from the use of this software.
//  Permission is granted to anyone to use this software for any purpose,
//  including commercial applications, and to alter it and redistribute it
//  freely, subject to the following restrictions:
//  1. The origin of this software must not be misrepresented; you must not
//     claim that you wrote the original software. If you use this software
//     in a product, an acknowledgment in the product documentation would be
//     appreciated but is not required.
//  2. Altered source versions must be plainly marked as such, and must not be
//     misrepresented as being the original software.
//  3. This notice may not be removed or altered from any source distribution.
//  (this is the zlib license)
use core::f32;

use aligned_vec::{avec, AVec, ConstAlign};

/// Fast exp approximation.
///
/// # Panics
///
/// If start of `xs` is not aligned to 32-bytes
pub fn exp(mut xs_aligned: AVec<f32, ConstAlign<32>>) -> AVec<f32, ConstAlign<32>> {
    const ALIGN: usize = 32;
    assert_eq!(xs_aligned.as_ptr() as usize % ALIGN, 0);
    let tail = if is_x86_feature_detected!("avx2") {
        let tail = xs_aligned.len() - (xs_aligned.len() % 8);
        unsafe {
            exp_avx2(&mut xs_aligned[..tail]);
        }
        tail
    } else {
        0
    };
    xs_aligned[tail..]
        .iter_mut()
        .for_each(|v| *v = exp_single(*v));
    xs_aligned
}

const EXP_HI: f32 = 88.37626;
const EXP_LO: f32 = -88.37626;
const P0: f32 = 1.9875691E-4;
const P1: f32 = 1.3981999E-3;
const P2: f32 = 8.333451E-3;
const P3: f32 = 4.1665795E-2;
const P4: f32 = 0.16666665;
const P5: f32 = 0.5;

fn exp_single(x: f32) -> f32 {
    let x = x.clamp(EXP_LO, EXP_HI);
    let fx = (x * f32::consts::LOG2_E).round();
    let z = fx * f32::consts::LN_2;
    let x = x - z;
    let z = x * x;
    let y = (((((P0 * x + P1) * x + P2) * x + P3) * x + P4) * x + P5) * z + x + 1.;
    // 2_f32.powf(fx);
    let fx = (fx as i32 + 127) << 23;
    y * f32::from_bits(fx as u32)
}

#[target_feature(enable = "avx2")]
unsafe fn exp_avx2(xs: &mut [f32]) {
    use std::arch::x86_64::*;
    let log2e = _mm256_set1_ps(f32::consts::LOG2_E);
    let ln2 = _mm256_set1_ps(f32::consts::LN_2);
    let one = _mm256_set1_ps(1.0);
    let exp_hi = _mm256_set1_ps(EXP_HI);
    let exp_lo = _mm256_set1_ps(EXP_LO);
    let p0 = _mm256_set1_ps(P0);
    let p1 = _mm256_set1_ps(P1);
    let p2 = _mm256_set1_ps(P2);
    let p3 = _mm256_set1_ps(P3);
    let p4 = _mm256_set1_ps(P4);
    let p5 = _mm256_set1_ps(P5);

    for i in (0..xs.len()).step_by(8) {
        let ptr = xs.as_mut_ptr().add(i);
        let x = _mm256_load_ps(ptr);
        let x = _mm256_min_ps(x, exp_hi);
        let x = _mm256_max_ps(x, exp_lo);

        let fx = _mm256_mul_ps(x, log2e);
        let fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let z = _mm256_mul_ps(fx, ln2);
        let x = _mm256_sub_ps(x, z);
        let z = _mm256_mul_ps(x, x);

        let y = p0;
        let y = _mm256_fmadd_ps(y, x, p1);
        let y = _mm256_fmadd_ps(y, x, p2);
        let y = _mm256_fmadd_ps(y, x, p3);
        let y = _mm256_fmadd_ps(y, x, p4);
        let y = _mm256_fmadd_ps(y, x, p5);
        let y = _mm256_fmadd_ps(y, z, x);
        let y = _mm256_add_ps(y, one);

        let imm0 = _mm256_cvttps_epi32(fx);
        let imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
        let imm0 = _mm256_slli_epi32(imm0, 23);
        let pow2n = _mm256_castsi256_ps(imm0);
        let y = _mm256_mul_ps(y, pow2n);
        _mm256_store_ps(ptr, y);
    }
}

#[test]
fn fast_exp_accurate() {
    let values: AVec<f32, ConstAlign<32>> =
        AVec::from_iter(32, (0..1024).map(|i| (i as f32 * 0.21) - 50.));
    let exps = exp(values.clone());
    println!("{exps:?}");
    for (v, e) in values.iter().zip(exps.iter()) {
        let expected = v.exp();
        let err = (expected - e).abs();
        assert!(err < 1e-3, "exp({v})={e}, actual {expected}, error={err}");
    }
}
