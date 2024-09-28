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
        target_feature = "fma",
        target_feature = "bmi1"
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
    target_feature = "fma",
    target_feature = "bmi1",
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
    const ACC_SIZE: usize = 32;
    let mut extr_flag_acc = [0u8; ACC_SIZE];
    // we read this acc as u64's
    static_assertions::const_assert_eq!(ACC_SIZE % 8, 0);
    // number of elements in `extr_flag_acc`
    let mut extr_flag_cnt = 0;
    // position in image corresponding to first bit in `extr_flag_acc`
    let mut extr_flag_acc_start = 0;

    #[rustfmt::skip]
    let masks: [_; 9] = [
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, i32::MIN), 
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, i32::MIN, i32::MIN), 
        _mm256_set_epi32(0, 0, 0, 0, 0, i32::MIN, i32::MIN, i32::MIN),
        _mm256_set_epi32(0, 0, 0, 0, i32::MIN, i32::MIN, i32::MIN, i32::MIN),
        _mm256_set_epi32(0, 0, 0, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN),
        _mm256_set_epi32(0, 0, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN),
        _mm256_set_epi32(0, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN),
        _mm256_set_epi32(i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN), 
    ];
    let mut write_mask;
    let mut load_mask;

    use std::arch::x86_64::*;

    let p0 = arr.as_slice().unwrap()[..(ny * nx)].as_ptr();
    let p1 = arr.as_slice().unwrap()[(ny * nx)..(2 * ny * nx)].as_ptr();
    let p2 = arr.as_slice().unwrap()[(2 * ny * nx)..(3 * ny * nx)].as_ptr();
    let ny = arr.shape()[1];
    let nx = arr.shape()[2];
    let mut extrema: Vec<RetainedExtremum> = Vec::default();
    let vvalue_threshold = _mm256_set1_ps(value_threshold);
    let sign_bit_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
    for y in (border)..(ny - border) {
        let mut x = border;
        let mut last_of_row = false;

        extr_flag_acc_start = border;
        write_mask = masks[8];
        load_mask = masks[8];
        while x < nx - border {
            if x + 8 >= nx - border {
                let dist_from_border = nx - border - x;
                //write_mask = masks[dist_from_border];
                write_mask = *masks.get_unchecked(dist_from_border);
                last_of_row = true;
                let dist_from_oob = nx - x;
                if dist_from_oob < 8 {
                    load_mask = masks[dist_from_oob];
                    //load_mask = *masks.get_unchecked(dist_from_oob);
                }
            }
            let val = _mm256_maskload_ps(p1.add(y * nx + x), load_mask);

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
            let v0 = _mm256_loadu_ps(p1.add(y * nx + x - 1)); // left
            let v1 = _mm256_loadu_ps(p1.add(y * nx + x + 1)); // right
            let v2 = _mm256_loadu_ps(p1.add((y - 1) * nx + x - 1)); // above left
            let v3 = _mm256_loadu_ps(p1.add((y - 1) * nx + x)); // above
            let v4 = _mm256_loadu_ps(p1.add((y - 1) * nx + x + 1)); // above right
            let v5 = _mm256_loadu_ps(p1.add((y + 1) * nx + x - 1)); // below left
            let v6 = _mm256_loadu_ps(p1.add((y + 1) * nx + x)); // below
            let v7 = _mm256_loadu_ps(p1.add((y + 1) * nx + x + 1)); // below right

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

            let v0 = _mm256_loadu_ps(p0.add(y * nx + x - 1)); // left
            let v1 = _mm256_loadu_ps(p0.add(y * nx + x));
            let v2 = _mm256_loadu_ps(p0.add(y * nx + x + 1)); // right
            let v3 = _mm256_loadu_ps(p0.add((y - 1) * nx + x - 1)); // above left
            let v4 = _mm256_loadu_ps(p0.add((y - 1) * nx + x)); // above
            let v5 = _mm256_loadu_ps(p0.add((y - 1) * nx + x + 1)); // above right
            let v6 = _mm256_loadu_ps(p0.add((y + 1) * nx + x - 1)); // below left
            let v7 = _mm256_loadu_ps(p0.add((y + 1) * nx + x)); // below
            let v8 = _mm256_loadu_ps(p0.add((y + 1) * nx + x + 1)); // below right

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

            let v0 = _mm256_loadu_ps(p2.add(y * nx + x - 1)); // left
            let v1 = _mm256_loadu_ps(p2.add(y * nx + x));
            let v2 = _mm256_loadu_ps(p2.add(y * nx + x + 1)); // right
            let v3 = _mm256_loadu_ps(p2.add((y - 1) * nx + x - 1)); // above left
            let v4 = _mm256_loadu_ps(p2.add((y - 1) * nx + x)); // above
            let v5 = _mm256_loadu_ps(p2.add((y - 1) * nx + x + 1)); // above right
            let v6 = _mm256_loadu_ps(p2.add((y + 1) * nx + x - 1)); // below left
            let v7 = _mm256_loadu_ps(p2.add((y + 1) * nx + x)); // below
            let v8 = _mm256_loadu_ps(p2.add((y + 1) * nx + x + 1)); // below right

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

            let is_extr = _mm256_and_ps(
                _mm256_or_ps(le_all, ge_all),
                _mm256_castsi256_ps(write_mask),
            );

            //println!(
            //    "{extr_flag_cnt}, {:08b}, x={x}, y={y}",
            //    _mm256_movemask_ps(is_extr).to_le_bytes()[0]
            //);
            *extr_flag_acc.get_unchecked_mut(extr_flag_cnt) =
                _mm256_movemask_ps(is_extr).to_le_bytes()[0];

            extr_flag_cnt += 1;
            if extr_flag_cnt == extr_flag_acc.len() || last_of_row {
                for i in (0..extr_flag_cnt).step_by(8) {
                    let mut qw = *(extr_flag_acc.as_ptr().add(i) as *const u64);
                    if qw == 0 {
                        continue;
                    }
                    //println!("i={i}");
                    //println!("acccnt={extr_flag_cnt}");
                    //println!("qwraw={qw:064b}");
                    if i + 8 == extr_flag_cnt && extr_flag_cnt % 8 != 0 {
                        assert!((extr_flag_cnt % 8) > 0, "{extr_flag_cnt}, {x}, {nx}");
                        qw >>= (8 - (extr_flag_cnt % 8)) * 8;
                    }
                    //println!("qw   ={qw:064b}");
                    let qw_start = (extr_flag_acc_start + i * 8) as u64;
                    while qw != 0 {
                        let trlz = _mm_tzcnt_64(qw) as u64;
                        let extr_x = qw_start + trlz;
                        //println!("qw   ={qw:064b}");
                        //println!("trlz={trlz}");
                        //if nx == 50 && scale == 1 && extr_x == 131 {
                        //    println!("y={y}, x={x}");
                        //    println!("qwstart={qw_start}, qw={qw:064b}");
                        //}
                        if let Some(extr) = discard_or_interpolate_extremum(
                            ScaleSpacePoint {
                                scale,
                                x: extr_x as usize,
                                y,
                            },
                            arr,
                            dog,
                        ) {
                            extrema.push(extr);
                        }
                        qw ^= 1 << trlz;
                        //println!("qw   ={qw:064b} after toggle bit");
                    }
                    //println!();
                }
                extr_flag_acc_start = x + 8;
                extr_flag_cnt = 0;
                extr_flag_acc.fill(0);
            }

            x += 8;
        }
    }
    extrema
}
