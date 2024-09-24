use ndarray::ArrayView3;

use crate::{discard_or_interpolate_extremum, RetainedExtremum, ScaleSpacePoint};

/// Finds extrema in `dogslice[1, :, :] i.e, values which are less or greater than`all their 26
/// neighbors in the 3D DoG.
pub fn local_extrema(
    dogslice: &ArrayView3<f32>,
    border: usize,
    value_threshold: f32,
    scale: usize,
    dog: &ArrayView3<f32>,
) -> Vec<RetainedExtremum> {
    assert!(border >= 1);
    assert!(2 * border <= dogslice.shape()[1]);
    assert!(2 * border <= dogslice.shape()[2]);
    let nz = dogslice.shape()[0];
    assert!(nz == 3);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        return unsafe { local_extrema_avx2(dogslice, border, value_threshold, scale, dog) };
    }
    local_extrema_fallback(dogslice, border, value_threshold, scale, dog)
}

fn local_extrema_fallback(
    arr: &ArrayView3<f32>,
    border: usize,
    value_threshold: f32,
    scale: usize,
    dog: &ArrayView3<f32>,
) -> Vec<RetainedExtremum> {
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
            if val.abs() <= value_threshold {
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
                if let Some(res) =
                    discard_or_interpolate_extremum(ScaleSpacePoint { scale, x, y }, arr, dog)
                {
                    extrema.push(res);
                }
            }
        }
    }
    extrema
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
unsafe fn local_extrema_avx2(
    arr: &ArrayView3<f32>,
    border: usize,
    value_threshold: f32,
    scale: usize,
    dog: &ArrayView3<f32>,
) -> Vec<RetainedExtremum> {
    assert!(border >= 1);
    assert!(2 * border <= arr.shape()[1]);
    assert!(2 * border <= arr.shape()[2]);
    let nx = arr.shape()[2];
    let ny = arr.shape()[1];
    let nz = arr.shape()[0];
    assert!(nz == 3);
    // TODO: MUST BE ALIGNED to alignof(u64)
    // bitflags marking extrema for 256 pixel positions
    let mut extr_flag_acc = [0u8; 32];
    // number of elements in `extr_flag_acc`
    let mut extr_flag_cnt = 0;
    // position in image corresponding to first bit in `extr_flag_acc`
    let mut extr_flac_acc_start = border;

    // if register of 8 f32's reaches outside of current image row, how many values are in image
    // bounds
    let mut use_values: usize;

    use std::arch::x86_64::*;
    let p0 = arr.as_slice().unwrap()[..(ny * nx)].as_ptr();
    let p1 = arr.as_slice().unwrap()[(ny * nx)..(2 * ny * nx)].as_ptr();
    let p2 = arr.as_slice().unwrap()[(2 * ny * nx)..(3 * ny * nx)].as_ptr();
    let ny = arr.shape()[1] as isize;
    let nx = arr.shape()[2] as isize;
    let mut extrema: Vec<RetainedExtremum> = Vec::default();
    let vvalue_threshold = _mm256_set1_ps(value_threshold);
    let sign_bit_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
    for y in (border as isize)..(ny - border as isize) {
        let mut x = border as isize;
        use_values = 0;
        let mut last_of_row = false;

        extr_flac_acc_start = border;
        while x < nx - border as isize {
            let dist_from_border = nx - border as isize - x;
            if dist_from_border < 8 {
                last_of_row = true;
                extr_flag_acc[extr_flag_cnt..].fill(0);
                x -= 8 - dist_from_border;
                use_values = dist_from_border as usize;
            }
            let val = _mm256_loadu_ps(p1.offset(y * nx + x));

            // zero out sign bit to take absolute value
            let abs = _mm256_and_ps(val, sign_bit_mask);
            let gt_thresh = _mm256_cmp_ps::<_CMP_GT_OQ>(abs, vvalue_threshold);

            macro_rules! mmax {
                ($a:expr, $b:expr) => {
                    _mm256_max_ps($a, $b)
                };
            }
            macro_rules! mmin {
                ($a:expr, $b:expr) => {
                    _mm256_min_ps($a, $b)
                };
            }
            let v0 = _mm256_loadu_ps(p1.offset(y * nx + x - 1)); // left
            let v1 = _mm256_loadu_ps(p1.offset(y * nx + x + 1)); // right
            let v2 = _mm256_loadu_ps(p1.offset((y - 1) * nx + x - 1)); // above left
            let v3 = _mm256_loadu_ps(p1.offset((y - 1) * nx + x)); // above
            let v4 = _mm256_loadu_ps(p1.offset((y - 1) * nx + x + 1)); // above right
            let v5 = _mm256_loadu_ps(p1.offset((y + 1) * nx + x - 1)); // below left
            let v6 = _mm256_loadu_ps(p1.offset((y + 1) * nx + x)); // below
            let v7 = _mm256_loadu_ps(p1.offset((y + 1) * nx + x + 1)); // below right

            // Tree shaped max computation to minimize dependencies between instructions
            let vmax = mmax!(
                mmax!(mmax!(v0, v1), mmax!(v2, v3)),
                mmax!(mmax!(v4, v5), mmax!(v6, v7))
            );
            let vmin = mmin!(
                mmin!(mmin!(v0, v1), mmin!(v2, v3)),
                mmin!(mmin!(v4, v5), mmin!(v6, v7))
            );
            let le_all = _mm256_and_ps(gt_thresh, _mm256_cmp_ps::<_CMP_LE_OQ>(val, vmin));
            let ge_all = _mm256_and_ps(gt_thresh, _mm256_cmp_ps::<_CMP_GE_OQ>(val, vmax));

            let v0 = _mm256_loadu_ps(p0.offset(y * nx + x - 1)); // left
            let v1 = _mm256_loadu_ps(p0.offset(y * nx + x));
            let v2 = _mm256_loadu_ps(p0.offset(y * nx + x + 1)); // right
            let v3 = _mm256_loadu_ps(p0.offset((y - 1) * nx + x - 1)); // above left
            let v4 = _mm256_loadu_ps(p0.offset((y - 1) * nx + x)); // above
            let v5 = _mm256_loadu_ps(p0.offset((y - 1) * nx + x + 1)); // above right
            let v6 = _mm256_loadu_ps(p0.offset((y + 1) * nx + x - 1)); // below left
            let v7 = _mm256_loadu_ps(p0.offset((y + 1) * nx + x)); // below
            let v8 = _mm256_loadu_ps(p0.offset((y + 1) * nx + x + 1)); // below right

            let vmax = mmax!(
                mmax!(mmax!(v0, v1), mmax!(v2, v3)),
                mmax!(mmax!(v4, v5), mmax!(v6, mmax!(v7, v8)))
            );
            let vmin = mmin!(
                mmin!(mmin!(v0, v1), mmin!(v2, v3)),
                mmin!(mmin!(v4, v5), mmin!(v6, mmin!(v7, v8)))
            );
            let le_all = _mm256_and_ps(le_all, _mm256_cmp_ps::<_CMP_LE_OQ>(val, vmin));
            let ge_all = _mm256_and_ps(ge_all, _mm256_cmp_ps::<_CMP_GE_OQ>(val, vmax));

            let v0 = _mm256_loadu_ps(p2.offset(y * nx + x - 1)); // left
            let v1 = _mm256_loadu_ps(p2.offset(y * nx + x));
            let v2 = _mm256_loadu_ps(p2.offset(y * nx + x + 1)); // right
            let v3 = _mm256_loadu_ps(p2.offset((y - 1) * nx + x - 1)); // above left
            let v4 = _mm256_loadu_ps(p2.offset((y - 1) * nx + x)); // above
            let v5 = _mm256_loadu_ps(p2.offset((y - 1) * nx + x + 1)); // above right
            let v6 = _mm256_loadu_ps(p2.offset((y + 1) * nx + x - 1)); // below left
            let v7 = _mm256_loadu_ps(p2.offset((y + 1) * nx + x)); // below
            let v8 = _mm256_loadu_ps(p2.offset((y + 1) * nx + x + 1)); // below right

            let vmax = mmax!(
                mmax!(mmax!(v0, v1), mmax!(v2, v3)),
                mmax!(mmax!(v4, v5), mmax!(v6, mmax!(v7, v8)))
            );
            let vmin = mmin!(
                mmin!(mmin!(v0, v1), mmin!(v2, v3)),
                mmin!(mmin!(v4, v5), mmin!(v6, mmin!(v7, v8)))
            );
            let le_all = _mm256_and_ps(le_all, _mm256_cmp_ps::<_CMP_LE_OQ>(val, vmin));
            let ge_all = _mm256_and_ps(ge_all, _mm256_cmp_ps::<_CMP_GE_OQ>(val, vmax));

            let is_extr = _mm256_or_ps(le_all, ge_all);

            extr_flag_acc[extr_flag_cnt] = _mm256_movemask_ps(is_extr).to_le_bytes()[0];
            extr_flag_cnt += 1;

            if last_of_row {
                extr_flag_acc[use_values..].fill(0)
            }
            if extr_flag_cnt == extr_flag_acc.len() - 1 || last_of_row {
                for i in (0..extr_flag_acc.len()).step_by(8) {
                    let mut qw = *(extr_flag_acc.as_ptr().add(i) as *const u64);
                    if qw == 0 {
                        continue;
                    }
                    let qw_start = (extr_flac_acc_start + i * 8) as u64;
                    while qw != 0 {
                        let trlz = _tzcnt_u64(qw);
                        let extr_x = qw_start + trlz;
                        if let Some(extr) = discard_or_interpolate_extremum(
                            ScaleSpacePoint {
                                scale,
                                x: extr_x as usize,
                                y: y as usize,
                            },
                            arr,
                            dog,
                        ) {
                            extrema.push(extr);
                        }
                        qw ^= 1 << trlz;
                    }
                }
                extr_flac_acc_start = x as usize + 8;
                extr_flag_cnt = 0;
            }

            x += 8;
        }
    }
    assert_eq!(
        extr_flac_acc_start + extr_flag_cnt,
        nx as usize - border as usize
    );
    extrema
}
