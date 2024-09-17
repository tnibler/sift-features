use ndarray::ArrayView3;

pub fn local_extrema(arr: &ArrayView3<f32>, border: usize, threshold: f32) -> Vec<(usize, usize)> {
    assert!(border >= 1);
    assert!(2 * border <= arr.shape()[1]);
    assert!(2 * border <= arr.shape()[2]);
    let nz = arr.shape()[0];
    assert!(nz == 3);
    #[cfg(target_arch = "x86_64")]
    {
        //if std::arch::is_x86_feature_detected!("avx2") {
        //    return unsafe { local_extrema_avx2(arr, border) };
        //}
    }
    local_extrema_fallback(arr, border, threshold)
}

#[target_feature(enable = "avx2")]
unsafe fn local_extrema_avx2(arr: &ArrayView3<f32>, border: usize) -> Vec<(usize, usize)> {
    assert!(border >= 1);
    assert!(2 * border <= arr.shape()[1]);
    assert!(2 * border <= arr.shape()[2]);
    let nx = arr.shape()[2];
    let ny = arr.shape()[1];
    let nz = arr.shape()[0];
    assert!(nz == 3);
    let mut scratch = vec![false; nx * 4];
    // a <= 0: a <= b => a - b <= 0
    // a >= 0: a >= b => a - b >= 0
    // a-b, shr 31,  not mask with sign bit
    use std::arch::x86_64::*;
    let p0 = arr.as_slice().unwrap()[..(ny * nx)].as_ptr();
    let p1 = arr.as_slice().unwrap()[(ny * nx)..(2 * ny * nx)].as_ptr();
    let p2 = arr.as_slice().unwrap()[(2 * ny * nx)..(3 * ny * nx)].as_ptr();
    let ny = arr.shape()[1] as isize;
    let nx = arr.shape()[2] as isize;
    let mut extrema: Vec<(usize, usize)> = Vec::default();
    let mut buf_idx: usize = 0;
    let mut buf_start: usize = (nx + 1) as usize;
    unsafe {
        for y in (border as isize)..(ny - border as isize) {
            let mut x = border as isize;

            if (buf_idx + nx as usize) >= scratch.len() {
                scratch
                    .iter()
                    .take(buf_idx)
                    .enumerate()
                    .for_each(|(idx, extr)| {
                        if *extr {
                            let ey = (buf_start + idx) / (nx as usize);
                            let ex = (buf_start + idx) % nx as usize;
                            extrema.push((ex, ey));
                        }
                    });
                buf_start = (y * nx + x) as usize;
                buf_idx = 0;
                scratch.fill(false);
            }
            while x + 8 < nx - border as isize {
                let mut sign0: __m256i = _mm256_setzero_si256();
                let mut sign1: __m256i = _mm256_set1_epi32(-1);
                let val = _mm256_loadu_ps(p1.offset(y * nx + x));
                macro_rules! do_sub {
                    ($p:expr, $offset:expr) => {
                        let other = _mm256_loadu_ps($p.offset($offset));
                        let sub = _mm256_sub_ps(val, other);
                        let sub: __m256i = std::mem::transmute::<__m256, __m256i>(sub);
                        let eqzero = _mm256_cmp_ps(val, other, 0);
                        sign1 = _mm256_and_si256(
                            _mm256_or_si256(sub, std::mem::transmute::<__m256, __m256i>(eqzero)),
                            sign1,
                        );
                        sign0 = _mm256_or_si256(sign0, sub);
                    };
                }
                do_sub!(p1, y * nx + x - 1); // left
                do_sub!(p1, y * nx + x + 1); // right
                do_sub!(p1, (y - 1) * nx + x - 1); // above left
                do_sub!(p1, (y - 1) * nx + x); // above
                do_sub!(p1, (y - 1) * nx + x + 1); // above right
                do_sub!(p1, (y + 1) * nx + x - 1); // below left
                do_sub!(p1, (y + 1) * nx + x); // below
                do_sub!(p1, (y + 1) * nx + x + 1); // below right

                do_sub!(p0, y * nx + x - 1); // left
                do_sub!(p0, y * nx + x);
                do_sub!(p0, y * nx + x + 1); // right
                do_sub!(p0, (y - 1) * nx + x - 1); // above left
                do_sub!(p0, (y - 1) * nx + x); // above
                do_sub!(p0, (y - 1) * nx + x + 1); // above right
                do_sub!(p0, (y + 1) * nx + x - 1); // below left
                do_sub!(p0, (y + 1) * nx + x); // below
                do_sub!(p0, (y + 1) * nx + x + 1); // below right

                do_sub!(p2, y * nx + x - 1); // left
                do_sub!(p2, y * nx + x);
                do_sub!(p2, y * nx + x + 1); // right
                do_sub!(p2, (y - 1) * nx + x - 1); // above left
                do_sub!(p2, (y - 1) * nx + x); // above
                do_sub!(p2, (y - 1) * nx + x + 1); // above right
                do_sub!(p2, (y + 1) * nx + x - 1); // below left
                do_sub!(p2, (y + 1) * nx + x); // below
                do_sub!(p2, (y + 1) * nx + x + 1); // below right

                // sign1 MSB is 1 iff val < others
                // sign0 MSB is 0 iff val > others
                // sign1 and val: MSB one if val < 0 and val < other => extremum
                // not(sign0 or val): MSB one if val > 0 and val > other => extremum
                let vs: __m256i = std::mem::transmute(val);
                let a = _mm256_and_si256(sign1, vs);
                let msb_one =
                    std::mem::transmute::<__m256i, __m256>(_mm256_set1_epi32(-0x80000000));
                let b = _mm256_xor_ps(
                    _mm256_or_ps(std::mem::transmute::<__m256i, __m256>(sign0), val),
                    msb_one,
                );
                let mask: __m256i =
                    std::mem::transmute(_mm256_or_si256(a, std::mem::transmute::<_, __m256i>(b)));
                let mask = _mm256_srli_epi32(mask, 31);

                let shuf = _mm256_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0,
                );

                let shuffled = _mm256_shuffle_epi8(mask, shuf);
                let packed = _mm256_permutevar8x32_epi32(
                    shuffled,
                    _mm256_set_epi32(-1, -1, -1, -1, -1, -1, 4, 0),
                );

                let out_loc = scratch.as_mut_ptr().add(buf_idx);
                let store_mask = _mm256_set_epi64x(0, 0, 0, i64::MIN);
                _mm256_maskstore_epi64(out_loc as *mut i64, store_mask, packed);

                x += 8;
                buf_idx += 8;
            }
            let y = y as usize;
            while x < nx - border as isize {
                let xx = x as usize;
                let yy = y as usize;
                let val = arr.uget((1, yy, xx));
                let c = if val >= &0. {
                    val >= arr.uget((1, yy, xx - 1))
                        && val >= arr.uget((1, yy, xx + 1))
                        && val >= arr.uget((1, yy + 1, xx + 1))
                        && val >= arr.uget((1, yy + 1, xx))
                        && val >= arr.uget((1, yy + 1, xx - 1))
                        && val >= arr.uget((1, yy - 1, xx + 1))
                        && val >= arr.uget((1, yy - 1, xx))
                        && val >= arr.uget((1, yy - 1, xx - 1))
                        && val >= arr.uget((0, yy, xx - 1))
                        && val >= arr.uget((0, yy, xx))
                        && val >= arr.uget((0, yy, xx + 1))
                        && val >= arr.uget((0, yy + 1, xx + 1))
                        && val >= arr.uget((0, yy + 1, xx))
                        && val >= arr.uget((0, yy + 1, xx - 1))
                        && val >= arr.uget((0, yy - 1, xx + 1))
                        && val >= arr.uget((0, yy - 1, xx))
                        && val >= arr.uget((0, yy - 1, xx - 1))
                        && val >= arr.uget((2, yy, xx - 1))
                        && val >= arr.uget((2, yy, xx))
                        && val >= arr.uget((2, yy, xx + 1))
                        && val >= arr.uget((2, yy + 1, xx + 1))
                        && val >= arr.uget((2, yy + 1, xx))
                        && val >= arr.uget((2, yy + 1, xx - 1))
                        && val >= arr.uget((2, yy - 1, xx + 1))
                        && val >= arr.uget((2, yy - 1, xx))
                        && val >= arr.uget((2, yy - 1, xx - 1))
                } else {
                    val <= arr.uget((1, yy, xx - 1))
                        && val <= arr.uget((1, yy, xx + 1))
                        && val <= arr.uget((1, yy + 1, xx + 1))
                        && val <= arr.uget((1, yy + 1, xx))
                        && val <= arr.uget((1, yy + 1, xx - 1))
                        && val <= arr.uget((1, yy - 1, xx + 1))
                        && val <= arr.uget((1, yy - 1, xx))
                        && val <= arr.uget((1, yy - 1, xx - 1))
                        && val <= arr.uget((0, yy, xx - 1))
                        && val <= arr.uget((0, yy, xx))
                        && val <= arr.uget((0, yy, xx + 1))
                        && val <= arr.uget((0, yy + 1, xx + 1))
                        && val <= arr.uget((0, yy + 1, xx))
                        && val <= arr.uget((0, yy + 1, xx - 1))
                        && val <= arr.uget((0, yy - 1, xx + 1))
                        && val <= arr.uget((0, yy - 1, xx))
                        && val <= arr.uget((0, yy - 1, xx - 1))
                        && val <= arr.uget((2, yy, xx - 1))
                        && val <= arr.uget((2, yy, xx))
                        && val <= arr.uget((2, yy, xx + 1))
                        && val <= arr.uget((2, yy + 1, xx + 1))
                        && val <= arr.uget((2, yy + 1, xx))
                        && val <= arr.uget((2, yy + 1, xx - 1))
                        && val <= arr.uget((2, yy - 1, xx + 1))
                        && val <= arr.uget((2, yy - 1, xx))
                        && val <= arr.uget((2, yy - 1, xx - 1))
                };
                scratch[buf_idx] = c;
                buf_idx += 1;
                x += 1;
            }
            buf_idx += 2 * border;
        }
    }
    scratch
        .iter()
        .take(buf_idx)
        .enumerate()
        .for_each(|(idx, extr)| {
            if *extr {
                let ey = (buf_start + idx) / (nx as usize);
                let ex = (buf_start + idx) - ey * nx as usize;
                extrema.push((ex, ey));
            }
        });
    extrema
}

fn local_extrema_fallback(
    arr: &ArrayView3<f32>,
    border: usize,
    threshold: f32,
) -> Vec<(usize, usize)> {
    assert!(border >= 1);
    assert!(2 * border <= arr.shape()[1]);
    assert!(2 * border <= arr.shape()[2]);
    let ny = arr.shape()[1];
    let nx = arr.shape()[2];
    let mut extrema = Vec::default();
    let nz = arr.shape()[0];
    assert!(nz == 3);
    for y in border..(ny - border) {
        for x in border..(nx - border) {
            let val = unsafe { arr.uget((1, y, x)) };
            if val.abs() <= threshold {
                continue;
            }
            let c = if val >= &0. {
                unsafe {
                    val >= arr.uget((1, y, x - 1))
                        && val >= arr.uget((1, y, x + 1))
                        && val >= arr.uget((1, y + 1, x + 1))
                        && val >= arr.uget((1, y + 1, x))
                        && val >= arr.uget((1, y + 1, x - 1))
                        && val >= arr.uget((1, y - 1, x + 1))
                        && val >= arr.uget((1, y - 1, x))
                        && val >= arr.uget((1, y - 1, x - 1))
                        && val >= arr.uget((0, y, x - 1))
                        && val >= arr.uget((0, y, x))
                        && val >= arr.uget((0, y, x + 1))
                        && val >= arr.uget((0, y + 1, x + 1))
                        && val >= arr.uget((0, y + 1, x))
                        && val >= arr.uget((0, y + 1, x - 1))
                        && val >= arr.uget((0, y - 1, x + 1))
                        && val >= arr.uget((0, y - 1, x))
                        && val >= arr.uget((0, y - 1, x - 1))
                        && val >= arr.uget((2, y, x - 1))
                        && val >= arr.uget((2, y, x))
                        && val >= arr.uget((2, y, x + 1))
                        && val >= arr.uget((2, y + 1, x + 1))
                        && val >= arr.uget((2, y + 1, x))
                        && val >= arr.uget((2, y + 1, x - 1))
                        && val >= arr.uget((2, y - 1, x + 1))
                        && val >= arr.uget((2, y - 1, x))
                        && val >= arr.uget((2, y - 1, x - 1))
                }
            } else {
                unsafe {
                    val <= arr.uget((1, y, x - 1))
                        && val <= arr.uget((1, y, x + 1))
                        && val <= arr.uget((1, y + 1, x + 1))
                        && val <= arr.uget((1, y + 1, x))
                        && val <= arr.uget((1, y + 1, x - 1))
                        && val <= arr.uget((1, y - 1, x + 1))
                        && val <= arr.uget((1, y - 1, x))
                        && val <= arr.uget((1, y - 1, x - 1))
                        && val <= arr.uget((0, y, x - 1))
                        && val <= arr.uget((0, y, x))
                        && val <= arr.uget((0, y, x + 1))
                        && val <= arr.uget((0, y + 1, x + 1))
                        && val <= arr.uget((0, y + 1, x))
                        && val <= arr.uget((0, y + 1, x - 1))
                        && val <= arr.uget((0, y - 1, x + 1))
                        && val <= arr.uget((0, y - 1, x))
                        && val <= arr.uget((0, y - 1, x - 1))
                        && val <= arr.uget((2, y, x - 1))
                        && val <= arr.uget((2, y, x))
                        && val <= arr.uget((2, y, x + 1))
                        && val <= arr.uget((2, y + 1, x + 1))
                        && val <= arr.uget((2, y + 1, x))
                        && val <= arr.uget((2, y + 1, x - 1))
                        && val <= arr.uget((2, y - 1, x + 1))
                        && val <= arr.uget((2, y - 1, x))
                        && val <= arr.uget((2, y - 1, x - 1))
                }
            };
            if c {
                extrema.push((x, y));
            }
        }
    }
    extrema
}
