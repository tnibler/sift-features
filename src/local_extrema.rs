use ndarray::ArrayView3;

pub fn local_extrema<T, CallerData: Copy>(
    arr: &ArrayView3<f32>,
    border: usize,
    value_threshold: f32,
    discard_or_process: impl Fn(usize, usize, CallerData, &mut Vec<T>),
    caller_data: CallerData,
) -> Vec<T> {
    assert!(border >= 1);
    assert!(2 * border <= arr.shape()[1]);
    assert!(2 * border <= arr.shape()[2]);
    let nz = arr.shape()[0];
    assert!(nz == 3);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                local_extrema_avx2(
                    arr,
                    border,
                    value_threshold,
                    discard_or_process,
                    caller_data,
                )
            };
        }
    }
    local_extrema_fallback(
        arr,
        border,
        value_threshold,
        discard_or_process,
        caller_data,
    )
}

pub fn local_extrema_fallback<T, CallerData: Copy>(
    arr: &ArrayView3<f32>,
    border: usize,
    threshold: f32,
    discard_or_process: impl Fn(usize, usize, CallerData, &mut Vec<T>),
    caller_data: CallerData,
) -> Vec<T> {
    assert!(border >= 1);
    assert!(2 * border <= arr.shape()[1]);
    assert!(2 * border <= arr.shape()[2]);

    let nz = arr.shape()[0];
    assert!(nz == 3);
    let ny = arr.shape()[1];
    let nx = arr.shape()[2];

    // Little bit of a dance to eliminate bounds checks and generate nice code
    let sl = arr.as_slice().expect("should be contiguous");
    let (p0, sl) = sl.split_at(nx * ny);
    let (p1, p2) = sl.split_at(nx * ny);
    let mut extrema = Vec::default();

    for y in border..(ny - border) {
        for x in border..(nx - border) {
            let val = p1[(y) * nx + x];
            if val.abs() <= threshold {
                continue;
            }
            let c = if val >= 0. {
                val >= p1[(y) * nx + x - 1]
                    && val >= p1[(y) * nx + x + 1]
                    && val >= p1[(y + 1) * nx + x + 1]
                    && val >= p1[(y + 1) * nx + x]
                    && val >= p1[(y + 1) * nx + x - 1]
                    && val >= p1[(y - 1) * nx + x + 1]
                    && val >= p1[(y - 1) * nx + x]
                    && val >= p1[(y - 1) * nx + x - 1]
                    && val >= p0[(y) * nx + x - 1]
                    && val >= p0[(y) * nx + x]
                    && val >= p0[(y) * nx + x + 1]
                    && val >= p0[(y + 1) * nx + x + 1]
                    && val >= p0[(y + 1) * nx + x]
                    && val >= p0[(y + 1) * nx + x - 1]
                    && val >= p0[(y - 1) * nx + x + 1]
                    && val >= p0[(y - 1) * nx + x]
                    && val >= p0[(y - 1) * nx + x - 1]
                    && val >= p2[(y) * nx + x - 1]
                    && val >= p2[(y) * nx + x]
                    && val >= p2[(y) * nx + x + 1]
                    && val >= p2[(y + 1) * nx + x + 1]
                    && val >= p2[(y + 1) * nx + x]
                    && val >= p2[(y + 1) * nx + x - 1]
                    && val >= p2[(y - 1) * nx + x + 1]
                    && val >= p2[(y - 1) * nx + x]
                    && val >= p2[(y - 1) * nx + x - 1]
            } else {
                val <= p1[(y) * nx + x - 1]
                    && val <= p1[(y) * nx + x + 1]
                    && val <= p1[(y + 1) * nx + x + 1]
                    && val <= p1[(y + 1) * nx + x]
                    && val <= p1[(y + 1) * nx + x - 1]
                    && val <= p1[(y - 1) * nx + x + 1]
                    && val <= p1[(y - 1) * nx + x]
                    && val <= p1[(y - 1) * nx + x - 1]
                    && val <= p0[(y) * nx + x - 1]
                    && val <= p0[(y) * nx + x]
                    && val <= p0[(y) * nx + x + 1]
                    && val <= p0[(y + 1) * nx + x + 1]
                    && val <= p0[(y + 1) * nx + x]
                    && val <= p0[(y + 1) * nx + x - 1]
                    && val <= p0[(y - 1) * nx + x + 1]
                    && val <= p0[(y - 1) * nx + x]
                    && val <= p0[(y - 1) * nx + x - 1]
                    && val <= p2[(y) * nx + x - 1]
                    && val <= p2[(y) * nx + x]
                    && val <= p2[(y) * nx + x + 1]
                    && val <= p2[(y + 1) * nx + x + 1]
                    && val <= p2[(y + 1) * nx + x]
                    && val <= p2[(y + 1) * nx + x - 1]
                    && val <= p2[(y - 1) * nx + x + 1]
                    && val <= p2[(y - 1) * nx + x]
                    && val <= p2[(y - 1) * nx + x - 1]
            };
            if c {
                discard_or_process(x, y, caller_data, &mut extrema);
            }
        }
    }
    extrema
}

#[target_feature(enable = "avx2")]
unsafe fn local_extrema_avx2<T, CallerData: Copy>(
    arr: &ArrayView3<f32>,
    border: usize,
    value_threshold: f32,
    discard_or_process: impl Fn(usize, usize, CallerData, &mut Vec<T>),
    caller_data: CallerData,
) -> Vec<T> {
    assert!(border >= 1);
    assert!(2 * border <= arr.shape()[1]);
    assert!(2 * border <= arr.shape()[2]);
    let nx = arr.shape()[2];
    let ny = arr.shape()[1];
    let nz = arr.shape()[0];
    assert!(nz == 3);
    let mut scratch = vec![false; 4 * 1024];
    // a <= 0: a <= b => a - b <= 0
    // a >= 0: a >= b => a - b >= 0
    // a-b, shr 31,  not mask with sign bit
    use std::arch::x86_64::*;
    let p0 = arr.as_slice().unwrap()[..(ny * nx)].as_ptr();
    let p1 = arr.as_slice().unwrap()[(ny * nx)..(2 * ny * nx)].as_ptr();
    let p2 = arr.as_slice().unwrap()[(2 * ny * nx)..(3 * ny * nx)].as_ptr();
    let ny = arr.shape()[1] as isize;
    let nx = arr.shape()[2] as isize;
    let mut extrema: Vec<T> = Vec::default();
    let mut buf_idx: usize = 0;
    let mut buf_start: usize = (nx as usize * border) + border;
    let vvalue_threshold = _mm256_set1_ps(value_threshold);
    let sign_bit_mask: __m256 = std::mem::transmute(_mm256_set1_epi32(i32::MAX));
    unsafe {
        for y in (border as isize)..(ny - border as isize) {
            let mut x = border as isize;

            while x + 8 < nx - border as isize {
                if (buf_idx + nx as usize) >= scratch.len() {
                    scratch
                        .iter()
                        .take(buf_idx)
                        .enumerate()
                        .for_each(|(idx, extr)| {
                            if *extr {
                                let ey = (buf_start + idx) / (nx as usize);
                                let ex = (buf_start + idx) % nx as usize; // TODO: replace modulo
                                discard_or_process(ex, ey, caller_data, &mut extrema);
                            }
                        });
                    buf_start = (y * nx + x) as usize;
                    buf_idx = 0;
                    scratch.fill(false);
                }
                let val = _mm256_loadu_ps(p1.offset(y * nx + x));

                // zero out sign bit to take absolute value
                let abs = _mm256_and_ps(val, sign_bit_mask);
                let lt_thresh = _mm256_cmp_ps::<_CMP_LT_OQ>(abs, vvalue_threshold);
                // MSB/sign bit is 0 iff val >= x forall x around val
                // Initialized to lt_thresh, so if abs(val) < threshold, MSB will always be set
                let mut sign0: __m256i = std::mem::transmute(lt_thresh);
                // MSB/sign bit is 1 iff val <= x forall x around val
                let mut sign1: __m256i = _mm256_set1_epi32(-1);
                macro_rules! do_sub {
                    ($p:expr, $offset:expr) => {
                        let other = _mm256_loadu_ps($p.offset($offset));
                        let sub = _mm256_sub_ps(val, other);
                        let sub: __m256i = std::mem::transmute::<__m256, __m256i>(sub);
                        let eqzero = _mm256_cmp_ps::<_CMP_EQ_OQ>(val, other);
                        // if sign bit of sub is not set and sub != 0, sign1 MSB will never go back
                        // to 1 and we know val is not >= x forall x around val
                        sign1 = _mm256_and_si256(
                            _mm256_or_si256(sub, std::mem::transmute::<__m256, __m256i>(eqzero)),
                            sign1,
                        );
                        // if sign bit of sub is set, sign0 MSB will never go back to 0 and we know
                        // val is not <= x forall x around val
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

                // sign1 MSB is 1 iff val <= other forall other
                // sign0 MSB is 0 iff val >= other forall other
                // sign1 and val: MSB 1 if val < 0 and val <= other => extremum
                // not(sign0 or val): MSB 1 if val > 0 and val >= other => extremum
                let vali: __m256i = std::mem::transmute(val);
                let neg_and_smaller = _mm256_and_si256(sign1, vali);
                let msb_one =
                    std::mem::transmute::<__m256i, __m256>(_mm256_set1_epi32(-0x80000000));
                let pos_and_larger = _mm256_xor_ps(
                    _mm256_or_ps(std::mem::transmute::<__m256i, __m256>(sign0), val),
                    msb_one,
                );
                let mask: __m256i = std::mem::transmute(_mm256_or_si256(
                    neg_and_smaller,
                    std::mem::transmute::<_, __m256i>(pos_and_larger),
                ));
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
                let val = arr.uget((1, y, xx));
                let c = if val >= &0. {
                    *val > value_threshold
                        && val >= arr.uget((1, y, xx - 1))
                        && val >= arr.uget((1, y, xx + 1))
                        && val >= arr.uget((1, y + 1, xx + 1))
                        && val >= arr.uget((1, y + 1, xx))
                        && val >= arr.uget((1, y + 1, xx - 1))
                        && val >= arr.uget((1, y - 1, xx + 1))
                        && val >= arr.uget((1, y - 1, xx))
                        && val >= arr.uget((1, y - 1, xx - 1))
                        && val >= arr.uget((0, y, xx - 1))
                        && val >= arr.uget((0, y, xx))
                        && val >= arr.uget((0, y, xx + 1))
                        && val >= arr.uget((0, y + 1, xx + 1))
                        && val >= arr.uget((0, y + 1, xx))
                        && val >= arr.uget((0, y + 1, xx - 1))
                        && val >= arr.uget((0, y - 1, xx + 1))
                        && val >= arr.uget((0, y - 1, xx))
                        && val >= arr.uget((0, y - 1, xx - 1))
                        && val >= arr.uget((2, y, xx - 1))
                        && val >= arr.uget((2, y, xx))
                        && val >= arr.uget((2, y, xx + 1))
                        && val >= arr.uget((2, y + 1, xx + 1))
                        && val >= arr.uget((2, y + 1, xx))
                        && val >= arr.uget((2, y + 1, xx - 1))
                        && val >= arr.uget((2, y - 1, xx + 1))
                        && val >= arr.uget((2, y - 1, xx))
                        && val >= arr.uget((2, y - 1, xx - 1))
                } else {
                    *val < value_threshold
                        && val <= arr.uget((1, y, xx - 1))
                        && val <= arr.uget((1, y, xx + 1))
                        && val <= arr.uget((1, y + 1, xx + 1))
                        && val <= arr.uget((1, y + 1, xx))
                        && val <= arr.uget((1, y + 1, xx - 1))
                        && val <= arr.uget((1, y - 1, xx + 1))
                        && val <= arr.uget((1, y - 1, xx))
                        && val <= arr.uget((1, y - 1, xx - 1))
                        && val <= arr.uget((0, y, xx - 1))
                        && val <= arr.uget((0, y, xx))
                        && val <= arr.uget((0, y, xx + 1))
                        && val <= arr.uget((0, y + 1, xx + 1))
                        && val <= arr.uget((0, y + 1, xx))
                        && val <= arr.uget((0, y + 1, xx - 1))
                        && val <= arr.uget((0, y - 1, xx + 1))
                        && val <= arr.uget((0, y - 1, xx))
                        && val <= arr.uget((0, y - 1, xx - 1))
                        && val <= arr.uget((2, y, xx - 1))
                        && val <= arr.uget((2, y, xx))
                        && val <= arr.uget((2, y, xx + 1))
                        && val <= arr.uget((2, y + 1, xx + 1))
                        && val <= arr.uget((2, y + 1, xx))
                        && val <= arr.uget((2, y + 1, xx - 1))
                        && val <= arr.uget((2, y - 1, xx + 1))
                        && val <= arr.uget((2, y - 1, xx))
                        && val <= arr.uget((2, y - 1, xx - 1))
                };
                scratch[buf_idx] = c;
                buf_idx += 1;
                x += 1;
            }
            buf_idx += 2 * border;
        }
    }
    buf_idx -= border;
    assert_eq!(buf_start + buf_idx, (nx as usize) * (ny as usize - border));
    scratch
        .iter()
        .take(buf_idx)
        .enumerate()
        .for_each(|(idx, extr)| {
            if *extr {
                let ey = (buf_start + idx) / (nx as usize);
                let ex = (buf_start + idx) - ey * nx as usize;
                discard_or_process(ex, ey, caller_data, &mut extrema);
            }
        });
    extrema
}
