use core::f32;
use std::cmp::{max, min};

use aligned_vec::{avec, AVec, ConstAlign};
use itertools::{izip, Itertools};
use ndarray::{s, Array3, ArrayView2, ArrayView3, ArrayViewMut3};

use crate::{
    atan2,
    exp::{self, exp},
    DESCRIPTOR_N_BINS, DESCRIPTOR_N_HISTOGRAMS, DESCRIPTOR_SIZE, LAMBDA_DESCR,
};

const BIN_ANGLE_STEP: f32 = DESCRIPTOR_N_BINS as f32 / 360.0;

pub fn compute_descriptor(
    img: &ArrayView2<f32>,
    x: f32,
    y: f32,
    scale: f32,
    orientation: f32,
    out: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    let hist = unsafe { compute_descriptor_avx2(img, x, y, scale, orientation, out) };
    let hist = ArrayView3::from_shape(
        (
            DESCRIPTOR_N_HISTOGRAMS + 2,
            DESCRIPTOR_N_HISTOGRAMS + 2,
            DESCRIPTOR_N_BINS,
        ),
        &hist,
    )
    .expect("shapes match");

    #[allow(clippy::reversed_empty_ranges)]
    let mut hist_flat = hist.slice_move(s![1..-1, 1..-1, ..]).to_owned();
    let hist_sl = hist_flat.as_slice().expect("array is flat");

    const DESCRIPTOR_MAGNITUDE_CAP: f32 = 0.2;
    let mut l2_sq = 0.0;
    hist_flat.iter().for_each(|x| l2_sq += x.powi(2));
    let l2_uncapped = hist_sl
        .chunks(4)
        .map(|xs| xs.iter().map(|x| x.powi(2)).sum())
        .reduce(|acc: f32, xs| acc + xs)
        .expect("array is not empty")
        .sqrt();
    let component_cap = l2_uncapped * DESCRIPTOR_MAGNITUDE_CAP;

    // Components of the vector can not be larger than 0.2 * l2_norm
    hist_flat.mapv_inplace(|v| v.min(component_cap));

    let l2_capped = hist_flat
        .iter()
        .copied()
        .chunks(4)
        .into_iter()
        .map(|xs| xs.into_iter().map(|x| x.powi(2)).sum())
        .reduce(|acc: f32, xs| acc + xs)
        .expect("array is not empty")
        .sqrt();

    const DESCRIPTOR_L2_NORM: f32 = 512.0;
    let l2_normalizer = DESCRIPTOR_L2_NORM / l2_capped.max(f32::EPSILON);

    // Saturating cast to u8
    hist_flat
        .into_iter()
        .map(move |x| {
            let x = (x * l2_normalizer).round() as i32;
            if x > (u8::MAX as i32) {
                u8::MAX
            } else {
                x as u8
            }
        })
        .zip(out.iter_mut())
        .for_each(|(hist_el, out_el)| *out_el = hist_el);
}

unsafe fn compute_descriptor_avx2(
    img: &ArrayView2<f32>,
    x: f32,
    y: f32,
    scale: f32,
    orientation: f32,
    out: &mut [u8],
) -> AVec<f32, ConstAlign<32>> {
    use std::arch::x86_64::*;
    assert_eq!(DESCRIPTOR_SIZE, out.len());
    let nhist = DESCRIPTOR_N_HISTOGRAMS;
    let nbins = DESCRIPTOR_N_BINS;
    let height = img.shape()[0] as u32;
    let width = img.shape()[1] as u32;
    let x = x.round() as u32;
    let y = y.round() as u32;
    let hist_width = LAMBDA_DESCR * scale;
    let radius = (LAMBDA_DESCR * scale * 2_f32.sqrt() * (nhist + 1) as f32 * 0.5).round() as i32;
    let (sin_ori, cos_ori) = orientation.to_radians().sin_cos();
    let (sin_ori_scaled, cos_ori_scaled) = (sin_ori / hist_width, cos_ori / hist_width);

    let sin_ori_scaled = _mm256_set1_ps(sin_ori_scaled);
    let cos_ori_scaled = _mm256_set1_ps(cos_ori_scaled);
    let img_ptr = img.as_ptr();

    let masks: [_; 8] = [
        _mm256_set_epi32(
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
        ),
        _mm256_set_epi32(
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            0,
        ),
        _mm256_set_epi32(
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            0,
            0,
        ),
        _mm256_set_epi32(i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, 0, 0, 0),
        _mm256_set_epi32(i32::MIN, i32::MIN, i32::MIN, i32::MIN, 0, 0, 0, 0),
        _mm256_set_epi32(i32::MIN, i32::MIN, i32::MIN, 0, 0, 0, 0, 0),
        _mm256_set_epi32(i32::MIN, i32::MIN, 0, 0, 0, 0, 0, 0),
        _mm256_set_epi32(i32::MIN, 0, 0, 0, 0, 0, 0, 0),
    ];
    let nhist_plus_two = _mm256_set1_epi32(nhist as i32 + 2);
    let nhist_half = _mm256_set1_ps((nhist / 2) as f32);
    let nhist_plus_half = _mm256_set1_ps(nhist as f32 + 0.5);
    let deg_per_rad = _mm256_set1_ps(180. / f32::consts::PI);

    // buffers for histogram
    let mut hist = avec!([32] | 0f32; (DESCRIPTOR_N_HISTOGRAMS + 2) * (DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS);

    const IDX_OFFSETS: [usize; 8] = [
        0,
        0,
        DESCRIPTOR_N_BINS,
        DESCRIPTOR_N_BINS,
        (DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS,
        (DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS,
        (DESCRIPTOR_N_HISTOGRAMS + 3) * DESCRIPTOR_N_BINS,
        (DESCRIPTOR_N_HISTOGRAMS + 3) * DESCRIPTOR_N_BINS,
    ];
    let index_offsets = _mm256_set_epi32(
        ((DESCRIPTOR_N_HISTOGRAMS + 3) * DESCRIPTOR_N_BINS) as i32,
        ((DESCRIPTOR_N_HISTOGRAMS + 3) * DESCRIPTOR_N_BINS) as i32,
        ((DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS) as i32,
        ((DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS) as i32,
        DESCRIPTOR_N_BINS as i32,
        DESCRIPTOR_N_BINS as i32,
        0,
        0,
    );
    let onehalf = _mm256_set1_ps(0.5);
    let onef = _mm256_set1_ps(1.);
    let twof = _mm256_set1_ps(2.);
    let bin_angle_step = _mm256_set1_ps(BIN_ANGLE_STEP);
    let n_bins = _mm256_set1_epi32(DESCRIPTOR_N_BINS as i32);
    let mod_nbins_mask = _mm256_set1_epi32(DESCRIPTOR_N_BINS as i32 - 1);
    let mut buf: AVec<f32, ConstAlign<32>> = avec!([32]| 0.; 8 * 8);
    let (buf000, buf001, buf010, buf011, buf100, buf101, buf110, buf111) = {
        let (buf000, sl) = buf.as_mut_slice().split_at_mut(8);
        let (buf001, sl) = sl.split_at_mut(8);
        let (buf010, sl) = sl.split_at_mut(8);
        let (buf011, sl) = sl.split_at_mut(8);
        let (buf100, sl) = sl.split_at_mut(8);
        let (buf101, sl) = sl.split_at_mut(8);
        let (buf110, sl) = sl.split_at_mut(8);
        let (buf111, sl) = sl.split_at_mut(8);
        assert_eq!(sl.len(), 0);
        (
            buf000, buf001, buf010, buf011, buf100, buf101, buf110, buf111,
        )
    };

    // -radius<=i<=radius
    // y+i < height-1
    // i start: max(-radius, -y+1)

    let y_winend: i32 = min(radius, height as i32 - 1 - y as i32);
    let y_winstart: i32 = min(y_winend, max(-radius, -(y as i32) + 1));
    let x_winend: i32 = min(radius, width as i32 - 1 - x as i32);
    let x_winstart: i32 = min(x_winend, max(-radius, -(x as i32) + 1));
    //println!("rad={radius}, y={y}, height={height}");
    //println!("{y_winstart}-{y_winend}");

    let x_ramp = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let x_rampf = _mm256_set_ps(0., 1., 2., 3., 4., 5., 6., 7.);
    let mask_all = _mm256_set1_epi32(i32::MIN);
    for y_win in y_winstart..y_winend {
        let y_abs = y as i32 + y_win;
        let y_abs_ = y as i32 + y_win;
        assert!(y_abs > 0);
        assert!((y_abs as u32) < height - 1);
        let y_abs = _mm256_set1_epi32(y_abs);
        let y_winf = _mm256_set1_ps(y_win as f32);
        for x_win in (x_winstart..x_winend).step_by(8) {
            let x_abs = x as i32 + x_win;
            assert!(x_abs > 0);
            let x_abs = x_abs as u32;
            assert!((x_abs as u32) < width - 1);
            let mask = if (x_abs + 8) as i32 > x_winend + x as i32 {
                masks[(8 - (x_winend + x as i32 - x_abs as i32)) as usize]
            } else {
                mask_all
            };
            //println!("oobmask={:?}", std::mem::transmute::<_, [f32; 8]>(mask));
            let x_abs_ = x_abs;
            let x_abs = _mm256_add_epi32(x_ramp, _mm256_set1_epi32(x_abs as i32));
            let x_winf = _mm256_add_ps(x_rampf, _mm256_set1_ps(x_win as f32));

            let col_rotated = _mm256_fmsub_ps(
                x_winf,
                cos_ori_scaled,
                _mm256_mul_ps(y_winf, sin_ori_scaled),
            );
            let row_rotated = _mm256_fmadd_ps(
                x_winf,
                sin_ori_scaled,
                _mm256_mul_ps(y_winf, cos_ori_scaled),
            );
            let row_bin = _mm256_add_ps(row_rotated, nhist_half);
            let col_bin = _mm256_add_ps(col_rotated, nhist_half);
            //println!("rb={row_bin:?}");
            //println!("cb={col_bin:?}");
            // row_bin > -0.5
            let mask = _mm256_and_ps(
                _mm256_castsi256_ps(mask),
                _mm256_cmp_ps::<_CMP_GT_OQ>(row_bin, _mm256_set1_ps(-0.5)),
            );
            // row_bin < n_hist + 0.5
            let mask = _mm256_and_ps(mask, _mm256_cmp_ps::<_CMP_LT_OQ>(row_bin, nhist_plus_half));
            // col_bin > -0.5
            let mask = _mm256_and_ps(
                mask,
                _mm256_cmp_ps::<_CMP_GT_OQ>(col_bin, _mm256_set1_ps(-0.5)),
            );
            // col_bin < n_hist + 0.5
            let mask = _mm256_and_ps(mask, _mm256_cmp_ps::<_CMP_LT_OQ>(col_bin, nhist_plus_half));
            //println!("binmask={:?}", std::mem::transmute::<_, [f32; 8]>(mask));

            let width = _mm256_set1_epi32(width as i32);
            // TODO: less muls and instructions to compute indices, all we need is +/- width or 1
            let pix_idx = _mm256_add_epi32(_mm256_mullo_epi32(width, y_abs), x_abs);
            let dx = {
                let idx1 = _mm256_add_epi32(pix_idx, _mm256_set1_epi32(1));
                let idx2 = _mm256_sub_epi32(pix_idx, _mm256_set1_epi32(1));
                let v1 = _mm256_mask_i32gather_ps::<4>(_mm256_setzero_ps(), img_ptr, idx1, mask);
                let v2 = _mm256_mask_i32gather_ps::<4>(_mm256_setzero_ps(), img_ptr, idx2, mask);
                _mm256_sub_ps(v1, v2)
            };
            let dy = {
                let idx1 = _mm256_sub_epi32(pix_idx, width);
                let idx2 = _mm256_add_epi32(pix_idx, width);
                let v1 = _mm256_mask_i32gather_ps::<4>(_mm256_setzero_ps(), img_ptr, idx1, mask);
                let v2 = _mm256_mask_i32gather_ps::<4>(_mm256_setzero_ps(), img_ptr, idx2, mask);
                _mm256_sub_ps(v1, v2)
            };
            let grad_nonzero_mask = {
                let dx_nz = _mm256_cmp_ps::<_CMP_NEQ_OQ>(dx, _mm256_setzero_ps());
                let dy_nz = _mm256_cmp_ps::<_CMP_NEQ_OQ>(dy, _mm256_setzero_ps());
                _mm256_or_ps(dx_nz, dy_nz)
            };
            let mask = _mm256_and_ps(mask, grad_nonzero_mask);
            let weight = _mm256_fmadd_ps(
                col_rotated,
                col_rotated,
                _mm256_mul_ps(row_rotated, row_rotated),
            );
            let weight = _mm256_mul_ps(
                weight,
                // weight_scale
                _mm256_set1_ps(-2. / (DESCRIPTOR_N_HISTOGRAMS.pow(2) as f32)),
            );
            let weight = exp::exp_avx2(weight);

            let magsq = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy));
            let mag = _mm256_mul_ps(magsq, _mm256_rsqrt_ps(magsq));
            //let mag = _mm256_sqrt_ps(magsq); // TODO use approx bove
            let atan = atan2::atan2_avx2(dx, dy);
            let deg = {
                let deg = _mm256_mul_ps(atan, deg_per_rad);
                // use sign bit to blend between deg and deg+360.
                _mm256_blendv_ps(deg, _mm256_add_ps(deg, _mm256_set1_ps(360.)), deg)
            };
            let normalized_ori = _mm256_sub_ps(deg, _mm256_set1_ps(orientation));

            let row_bin = _mm256_sub_ps(row_bin, onehalf);
            let col_bin = _mm256_sub_ps(col_bin, onehalf);
            let mag = _mm256_mul_ps(mag, weight);
            let ori_bin = _mm256_mul_ps(normalized_ori, bin_angle_step);

            let row_floor = _mm256_floor_ps(row_bin);
            let col_floor = _mm256_floor_ps(col_bin);
            let ori_floor = _mm256_floor_ps(ori_bin);
            let row_frac = _mm256_sub_ps(row_bin, row_floor);
            let col_frac = _mm256_sub_ps(col_bin, col_floor);
            let ori_frac = _mm256_sub_ps(ori_bin, ori_floor);

            let c1 = _mm256_mul_ps(mag, row_frac);
            let c0 = _mm256_sub_ps(mag, c1);
            let c11 = _mm256_mul_ps(c1, col_frac);
            let c10 = _mm256_sub_ps(c1, c11);
            let c01 = _mm256_mul_ps(c0, col_frac);
            let c00 = _mm256_sub_ps(c0, c01);
            let c111 = _mm256_mul_ps(c11, ori_frac);
            let c110 = _mm256_sub_ps(c11, c111);
            let c101 = _mm256_mul_ps(c10, ori_frac);
            let c100 = _mm256_sub_ps(c10, c101);
            let c011 = _mm256_mul_ps(c01, ori_frac);
            let c010 = _mm256_sub_ps(c01, c011);
            let c001 = _mm256_mul_ps(c00, ori_frac);
            let c000 = _mm256_sub_ps(c00, c001);

            let mask = _mm256_castps_si256(mask);
            _mm256_storeu_ps(buf000.as_mut_ptr(), c000);
            _mm256_storeu_ps(buf001.as_mut_ptr(), c001);
            _mm256_storeu_ps(buf010.as_mut_ptr(), c010);
            _mm256_storeu_ps(buf011.as_mut_ptr(), c011);
            _mm256_storeu_ps(buf100.as_mut_ptr(), c100);
            _mm256_storeu_ps(buf101.as_mut_ptr(), c101);
            _mm256_storeu_ps(buf110.as_mut_ptr(), c110);
            _mm256_storeu_ps(buf111.as_mut_ptr(), c111);

            let row_floor_p1 = _mm256_cvttps_epi32(_mm256_add_ps(row_floor, onef));
            let col_floor_p1 = _mm256_cvttps_epi32(_mm256_add_ps(col_floor, onef));

            let ori_floor = _mm256_cvttps_epi32(ori_floor);

            //let row_floor_p1 = {
            //    let mut v = [0_i32; 8];
            //    _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, row_floor_p1);
            //    v
            //};
            //let col_floor_p1 = {
            //    let mut v = [0_i32; 8];
            //    _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, col_floor_p1);
            //    v
            //};
            let msk = {
                let mut v = [0_i32; 8];
                _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, mask);
                v
            };
            for j in 0..8 {
                if msk[j] == 0 {
                    continue;
                }
                let jj = _mm256_set1_epi32(j as i32);
                let ori_alt_ones = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
                // set every element to ori_floor[j]
                let row_floor_p1 = _mm256_permutevar8x32_epi32(row_floor_p1, jj);
                let col_floor_p1 = _mm256_permutevar8x32_epi32(col_floor_p1, jj);
                let ori_single = _mm256_permutevar8x32_epi32(ori_floor, jj);

                let ori_idx_offset =
                    _mm256_and_si256(_mm256_add_epi32(ori_single, ori_alt_ones), mod_nbins_mask);
                // TODO: check what's fater here
                let idx_base = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_mullo_epi32(row_floor_p1, nhist_plus_two), n_bins),
                    _mm256_mullo_epi32(col_floor_p1, n_bins),
                );
                //let idx_base = _mm256_set1_epi32(
                //    row_floor_p1[j] * (nhist as i32 + 2) * nbins as i32
                //        + col_floor_p1[j] * nbins as i32,
                //);
                let idx =
                    _mm256_add_epi32(_mm256_add_epi32(idx_base, index_offsets), ori_idx_offset);
                let idx = {
                    let mut v = [0_i32; 8];
                    _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, idx);
                    v
                };

                hist[idx[0] as usize] += buf000[j];
                hist[idx[1] as usize] += buf001[j];
                hist[idx[2] as usize] += buf010[j];
                hist[idx[3] as usize] += buf011[j];
                hist[idx[4] as usize] += buf100[j];
                hist[idx[5] as usize] += buf101[j];
                hist[idx[6] as usize] += buf110[j];
                hist[idx[7] as usize] += buf111[j];
            }
        }
    }
    hist
}

/// SAFETY: alignment 32, all same length, length divisible by 8, called if avx2 avail
#[target_feature(enable = "avx2")]
unsafe fn histogram_avx2(
    row_bins: &[f32],
    col_bins: &[f32],
    orientations: &[f32],
    magnitudes: &[f32],
    weights: &[f32],
    hist: &mut ArrayViewMut3<f32>,
) {
    use std::arch::x86_64::*;
    use std::mem::transmute;

    static_assertions::const_assert_eq!(DESCRIPTOR_N_BINS, 8);

    assert_eq!(
        hist.shape(),
        [
            DESCRIPTOR_N_HISTOGRAMS + 2,
            DESCRIPTOR_N_HISTOGRAMS + 2,
            DESCRIPTOR_N_BINS
        ]
    );
    let hist = hist.as_slice_mut().expect("must be contiguous");

    let nhist = DESCRIPTOR_N_HISTOGRAMS as i32;
    let nbins = DESCRIPTOR_N_BINS as i32;
    const IDX_OFFSETS: [usize; 8] = [
        0,
        0,
        DESCRIPTOR_N_BINS,
        DESCRIPTOR_N_BINS,
        (DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS,
        (DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS,
        (DESCRIPTOR_N_HISTOGRAMS + 3) * DESCRIPTOR_N_BINS,
        (DESCRIPTOR_N_HISTOGRAMS + 3) * DESCRIPTOR_N_BINS,
    ];
    let index_offsets = _mm256_set_epi32(
        ((DESCRIPTOR_N_HISTOGRAMS + 3) * DESCRIPTOR_N_BINS) as i32,
        ((DESCRIPTOR_N_HISTOGRAMS + 3) * DESCRIPTOR_N_BINS) as i32,
        ((DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS) as i32,
        ((DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS) as i32,
        DESCRIPTOR_N_BINS as i32,
        DESCRIPTOR_N_BINS as i32,
        0,
        0,
    );
    let onehalf = _mm256_set1_ps(0.5);
    let onef = _mm256_set1_ps(1.);
    let twof = _mm256_set1_ps(2.);
    let bin_angle_step = _mm256_set1_ps(BIN_ANGLE_STEP);
    let n_bins = _mm256_set1_ps(DESCRIPTOR_N_BINS as f32);
    let mod_nbins_mask = _mm256_set1_epi32(DESCRIPTOR_N_BINS as i32 - 1);
    let mut buf: AVec<f32, ConstAlign<32>> = avec!([32]| 0.; 8 * 8);
    let (buf000, buf001, buf010, buf011, buf100, buf101, buf110, buf111) = {
        let (buf000, sl) = buf.as_mut_slice().split_at_mut(8);
        let (buf001, sl) = sl.split_at_mut(8);
        let (buf010, sl) = sl.split_at_mut(8);
        let (buf011, sl) = sl.split_at_mut(8);
        let (buf100, sl) = sl.split_at_mut(8);
        let (buf101, sl) = sl.split_at_mut(8);
        let (buf110, sl) = sl.split_at_mut(8);
        let (buf111, sl) = sl.split_at_mut(8);
        assert_eq!(sl.len(), 0);
        (
            buf000, buf001, buf010, buf011, buf100, buf101, buf110, buf111,
        )
    };

    for i in (0_usize..row_bins.len()).step_by(8) {
        let i = i as isize;
        let row_bin = _mm256_load_ps(row_bins.as_ptr().offset(i));
        let col_bin = _mm256_load_ps(col_bins.as_ptr().offset(i));
        let mag = _mm256_load_ps(magnitudes.as_ptr().offset(i));
        let ori = _mm256_load_ps(orientations.as_ptr().offset(i));
        let weight = _mm256_load_ps(weights.as_ptr().offset(i));

        let row_bin = _mm256_sub_ps(row_bin, onehalf);
        let col_bin = _mm256_sub_ps(col_bin, onehalf);
        let mag = _mm256_mul_ps(mag, weight);
        let ori_bin = _mm256_mul_ps(ori, bin_angle_step);
        //println!("mag={mag:?}");

        let row_floor = _mm256_floor_ps(row_bin);
        let col_floor = _mm256_floor_ps(col_bin);
        let ori_floor = _mm256_floor_ps(ori_bin);
        let row_frac = _mm256_sub_ps(row_bin, row_floor);
        let col_frac = _mm256_sub_ps(col_bin, col_floor);
        let ori_frac = _mm256_sub_ps(ori_bin, ori_floor);

        let c1 = _mm256_mul_ps(mag, row_frac);
        let c0 = _mm256_sub_ps(mag, c1);
        let c11 = _mm256_mul_ps(c1, col_frac);
        let c10 = _mm256_sub_ps(c1, c11);
        let c01 = _mm256_mul_ps(c0, col_frac);
        let c00 = _mm256_sub_ps(c0, c01);
        let c111 = _mm256_mul_ps(c11, ori_frac);
        let c110 = _mm256_sub_ps(c11, c111);
        let c101 = _mm256_mul_ps(c10, ori_frac);
        let c100 = _mm256_sub_ps(c10, c101);
        let c011 = _mm256_mul_ps(c01, ori_frac);
        let c010 = _mm256_sub_ps(c01, c011);
        let c001 = _mm256_mul_ps(c00, ori_frac);
        let c000 = _mm256_sub_ps(c00, c001);

        _mm256_store_ps(buf000.as_mut_ptr(), c000);
        _mm256_store_ps(buf001.as_mut_ptr(), c001);
        _mm256_store_ps(buf010.as_mut_ptr(), c010);
        _mm256_store_ps(buf011.as_mut_ptr(), c011);
        _mm256_store_ps(buf100.as_mut_ptr(), c100);
        _mm256_store_ps(buf101.as_mut_ptr(), c101);
        _mm256_store_ps(buf110.as_mut_ptr(), c110);
        _mm256_store_ps(buf111.as_mut_ptr(), c111);

        let row_floor_p1 = _mm256_cvttps_epi32(_mm256_add_ps(row_floor, onef));
        let col_floor_p1 = _mm256_cvttps_epi32(_mm256_add_ps(col_floor, onef));

        let ori_floor = _mm256_cvttps_epi32(ori_floor);

        let row_floor_p1 = {
            let mut v = [0_i32; 8];
            _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, row_floor_p1);
            v
        };
        let col_floor_p1 = {
            let mut v = [0_i32; 8];
            _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, col_floor_p1);
            v
        };
        let nhist = nhist as usize;
        let nbins = nbins as usize;
        let mut permute = _mm256_setzero_si256();
        for j in 0..8 {
            let ori_p1s = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
            let ori_single = _mm256_permutevar8x32_epi32(ori_floor, permute);
            let ori_idx_offset =
                _mm256_and_si256(_mm256_add_epi32(ori_single, ori_p1s), mod_nbins_mask);
            let idx_base = _mm256_set1_epi32(
                row_floor_p1[j] * (nhist as i32 + 2) * nbins as i32
                    + col_floor_p1[j] * nbins as i32,
            );
            let idx = _mm256_add_epi32(_mm256_add_epi32(idx_base, index_offsets), ori_idx_offset);
            let idx = {
                let mut v = [0_i32; 8];
                _mm256_storeu_si256(transmute::<*mut i32, *mut __m256i>(v.as_mut_ptr()), idx);
                v
            };
            //println!("{idx:?}");
            //println!(
            //    "{}, {}, {}, {}, {}, {}, {}, {}",
            //    buf000[j],
            //    buf000[j],
            //    buf010[j],
            //    buf011[j],
            //    buf100[j],
            //    buf101[j],
            //    buf110[j],
            //    buf111[j],
            //);
            hist[idx[0] as usize] += buf000[j];
            hist[idx[1] as usize] += buf001[j];
            hist[idx[2] as usize] += buf010[j];
            hist[idx[3] as usize] += buf011[j];
            hist[idx[4] as usize] += buf100[j];
            hist[idx[5] as usize] += buf101[j];
            hist[idx[6] as usize] += buf110[j];
            hist[idx[7] as usize] += buf111[j];
            permute = _mm256_add_epi32(permute, _mm256_set1_epi32(1));
        }
    }
}
