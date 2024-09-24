use core::f32;
use std::cmp::{max, min};

use aligned_vec::{avec, AVec, ConstAlign};
use itertools::{izip, Itertools};
use ndarray::{s, ArrayView2, ArrayView3, ArrayViewMut3};

use crate::{atan2, DESCRIPTOR_N_BINS, DESCRIPTOR_N_HISTOGRAMS, LAMBDA_DESCR};

const BIN_ANGLE_STEP: f32 = DESCRIPTOR_N_BINS as f32 / 360.0;
const DESCRIPTOR_L2_NORM: f32 = 512.0;
const DESCRIPTOR_MAGNITUDE_CAP: f32 = 0.2;

pub fn compute_descriptor(
    img: &ArrayView2<f32>,
    x: f32,
    y: f32,
    scale: f32,
    orientation: f32,
    out: &mut [u8],
) {
    let hist = raw_descriptor(img, x, y, scale, orientation);
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

    let hist_sl = hist_flat.as_slice_mut().expect("array is flat");

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        unsafe { normalize_hist_avx2(hist_sl, out) };
        return;
    }

    // Fallback scalar implementation

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

#[inline(always)]
fn raw_descriptor(
    img: &ArrayView2<f32>,
    x: f32,
    y: f32,
    scale: f32,
    orientation: f32,
) -> AVec<f32, ConstAlign<32>> {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        return unsafe { raw_descriptor_avx2(img, x, y, scale, orientation) };
    }

    // Fallback scalar implemenation
    let n_hist = DESCRIPTOR_N_HISTOGRAMS;
    let n_bins = DESCRIPTOR_N_BINS;
    let height = img.shape()[0];
    let width = img.shape()[1];
    let x = x.round() as usize;
    let y = y.round() as usize;
    let hist_width = LAMBDA_DESCR * scale;
    let radius = (LAMBDA_DESCR * scale * 2_f32.sqrt() * (n_hist + 1) as f32 * 0.5).round() as i32;
    let (sin_ori, cos_ori) = orientation.to_radians().sin_cos();
    let (sin_ori_scaled, cos_ori_scaled) = (sin_ori / hist_width, cos_ori / hist_width);

    const ALIGN: usize = 32;
    let cap = (4 * radius * radius) as usize;
    let mut gradients_x: AVec<f32, ConstAlign<ALIGN>> = AVec::with_capacity(ALIGN, cap);
    let mut gradients_y: AVec<f32, ConstAlign<ALIGN>> = AVec::with_capacity(ALIGN, cap);
    let mut row_bins: AVec<f32, ConstAlign<ALIGN>> = AVec::with_capacity(ALIGN, cap);
    let mut col_bins: AVec<f32, ConstAlign<ALIGN>> = AVec::with_capacity(ALIGN, cap);
    let mut weights: AVec<f32, ConstAlign<ALIGN>> = AVec::with_capacity(ALIGN, cap);
    (-radius..=radius)
        .flat_map(|y_in_window| {
            (-radius..=radius).filter_map(move |x_in_window| {
                // row and col in the keypoint's coordinates wrt its reference orientation
                let col_rotated: f32 =
                    x_in_window as f32 * cos_ori_scaled - y_in_window as f32 * sin_ori_scaled;
                let row_rotated: f32 =
                    x_in_window as f32 * sin_ori_scaled + y_in_window as f32 * cos_ori_scaled;
                // Bin here means which of the 4*4 histograms the gradient at this point will
                // contribute to. It is not a bin within a histogram.
                let row_bin = row_rotated + (n_hist / 2) as f32;
                let col_bin = col_rotated + (n_hist / 2) as f32;

                // coordinates to read pixels from. No resampling here
                let abs_y = y as i32 + y_in_window;
                let abs_x = x as i32 + x_in_window;

                // +/- 0.5 to check if the sample would contribute anything to the 4*4 histograms
                // of interest with interpolation.
                if row_bin > -0.5
                    && row_bin < n_hist as f32 + 0.5
                    && col_bin > -0.5
                    && col_bin < n_hist as f32 + 0.5
                    && abs_y > 0
                    && abs_y < (height - 1) as i32
                    && abs_x > 0
                    && abs_x < (width - 1) as i32
                {
                    let abs_y = abs_y as usize;
                    let abs_x = abs_x as usize;
                    let dx = img[(abs_y, abs_x + 1)] - img[(abs_y, abs_x - 1)];
                    let dy = img[(abs_y - 1, abs_x)] - img[(abs_y + 1, abs_x)];

                    if dx == 0. && dy == 0. {
                        return None;
                    }

                    // Samples contribute less to histogram as they get further away.
                    // Exponents in Eq. (27) in [4]
                    let weight = col_rotated.powi(2) + row_rotated.powi(2);
                    Some((dx, dy, row_bin, col_bin, weight))
                } else {
                    None
                }
            })
        })
        .for_each(|(dx, dy, row_bin, col_bin, weight)| {
            gradients_x.push(dx);
            gradients_y.push(dy);
            row_bins.push(row_bin);
            col_bins.push(col_bin);
            weights.push(weight);
        });
    assert_eq!(gradients_x.len(), gradients_y.len());
    assert_eq!(gradients_x.len(), row_bins.len());
    assert_eq!(gradients_x.len(), col_bins.len());
    assert_eq!(gradients_x.len(), weights.len());

    // Different weighting than in [4]
    let weight_scale = -2. / (n_hist.pow(2) as f32);
    let weights: AVec<_, ConstAlign<ALIGN>> =
        AVec::from_iter(ALIGN, weights.iter().map(|x| (x * weight_scale).exp()));

    // Gradient orientations in patch normalized wrt to the keypoint's reference orientation.
    let normalized_orientations: AVec<_, ConstAlign<ALIGN>> = {
        let atans = {
            let mut gx = gradients_x.clone();
            atan2::atan2_inplace(&mut gx, &gradients_y);
            gx
        };
        AVec::from_iter(
            ALIGN,
            atans.iter().map(|angle| {
                let deg = angle.to_degrees();
                let deg = if deg < 0. { deg + 360.0 } else { deg };
                deg - orientation
            }),
        )
    };
    // Gradient magnitudes
    let magnitudes: AVec<_, ConstAlign<ALIGN>> = AVec::from_iter(
        ALIGN,
        gradients_x
            .iter()
            .zip(gradients_y.iter())
            .map(|(x, y)| (x * x + y * y).sqrt()),
    );

    // Instead of 4*4 histograms, we work with 5*5 here so that the interpolation works out simpler
    // at the borders (surely possible to do differently as well).
    // The outermost histograms will be discarded.
    let mut hist_buf: AVec<f32, ConstAlign<ALIGN>> =
        avec!([ALIGN] | 0.; (n_hist + 2) * (n_hist + 2) * DESCRIPTOR_N_BINS);
    let mut hist =
        ArrayViewMut3::from_shape((n_hist + 2, n_hist + 2, DESCRIPTOR_N_BINS), &mut hist_buf)
            .expect("shape matches");

    // Spread each sample point's contribution to its 8 neighbouring histograms based on its distance
    // from the histogram window's center and weighted by the sample's gradient magnitude.
    izip!(
        row_bins.iter(),
        col_bins.iter(),
        normalized_orientations.iter(),
        magnitudes.iter(),
        weights.iter()
    )
    .for_each(|(row_bin, col_bin, orientation, mag, weight)| {
        // Subtracting 0.5 here because the trilinear interpolation (the reverse actually)
        // below works on the {-0.5, 0.5}^3 cube, but our histograms are located in {0, 1}^ cubes.
        let row_bin = row_bin - 0.5;
        let col_bin = col_bin - 0.5;
        let mag = mag * weight;
        let ori_bin = orientation * BIN_ANGLE_STEP;
        let row_floor = row_bin.floor();
        let col_floor = col_bin.floor();
        let ori_floor = ori_bin.floor();
        let row_frac = row_bin - row_floor;
        let col_frac = col_bin - col_floor;
        let ori_frac = ori_bin - ori_floor;

        // The numbers are to be seen as coordinates on a cube.
        // Notation taken from https://en.wikipedia.org/wiki/Trilinear_interpolation.
        let c1 = mag * row_frac;
        let c0 = mag - c1;
        let c11 = c1 * col_frac;
        let c10 = c1 - c11;
        let c01 = c0 * col_frac;
        let c00 = c0 - c01;
        let c111 = c11 * ori_frac;
        let c110 = c11 - c111;
        let c101 = c10 * ori_frac;
        let c100 = c10 - c101;
        let c011 = c01 * ori_frac;
        let c010 = c01 - c011;
        let c001 = c00 * ori_frac;
        let c000 = c00 - c001;

        let row_floor_p1 = (row_floor + 1.) as usize;
        let col_floor_p1 = (col_floor + 1.) as usize;
        let row_floor_p2 = (row_floor + 2.) as usize;
        let col_floor_p2 = (col_floor + 2.) as usize;

        // Histogram bin indices wrap around because angles
        let ori_floor = if ori_floor < 0. {
            ori_floor + n_bins as f32
        } else if ori_floor >= n_bins as f32 {
            ori_floor - n_bins as f32
        } else {
            ori_floor
        } as usize;
        let ori_floor_p1 = if ori_floor + 1 >= n_bins {
            // wrap around to ori_floor + 1 - n_bins, can only be 0
            0
        } else {
            ori_floor + 1
        };

        hist[(row_floor_p1, col_floor_p1, ori_floor)] += c000;
        hist[(row_floor_p1, col_floor_p1, ori_floor_p1)] += c001;
        hist[(row_floor_p1, col_floor_p2, ori_floor)] += c010;
        hist[(row_floor_p1, col_floor_p2, ori_floor_p1)] += c011;
        hist[(row_floor_p2, col_floor_p1, ori_floor)] += c100;
        hist[(row_floor_p2, col_floor_p1, ori_floor_p1)] += c101;
        hist[(row_floor_p2, col_floor_p2, ori_floor)] += c110;
        hist[(row_floor_p2, col_floor_p2, ori_floor_p1)] += c111;
    });
    hist_buf
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
unsafe fn raw_descriptor_avx2(
    img: &ArrayView2<f32>,
    x: f32,
    y: f32,
    scale: f32,
    orientation: f32,
) -> AVec<f32, ConstAlign<32>> {
    use std::arch::x86_64::*;

    use crate::exp;
    let nhist = DESCRIPTOR_N_HISTOGRAMS;
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

    #[rustfmt::skip]
    let masks: [_; 8] = [
        _mm256_set_epi32(i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN), 
        _mm256_set_epi32(i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, 0), 
        _mm256_set_epi32(i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, 0, 0), 
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

    let mut hist = avec!([32] | 0f32; (DESCRIPTOR_N_HISTOGRAMS + 2) * (DESCRIPTOR_N_HISTOGRAMS + 2) * DESCRIPTOR_N_BINS);

    let mut tmp_row_floor_p1 = avec!([32] | 0; 8);
    let mut tmp_col_floor_p1 = avec!([32] | 0; 8);
    let mut tmp_ori_floor = avec!([32] | 0; 8);

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

    let y_winend: i32 = min(radius, height as i32 - 1 - y as i32);
    let y_winstart: i32 = min(y_winend, max(-radius, -(y as i32) + 1));
    let x_winend: i32 = min(radius, width as i32 - 1 - x as i32);
    let x_winstart: i32 = min(x_winend, max(-radius, -(x as i32) + 1));

    let x_ramp = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let x_rampf = _mm256_set_ps(0., 1., 2., 3., 4., 5., 6., 7.);
    let mask_all = _mm256_set1_epi32(i32::MIN);
    let mut pix_idx_rowstart = _mm256_add_epi32(
        _mm256_set1_epi32((y as i32 + y_winstart) * width as i32 + x as i32 + x_winstart),
        x_ramp,
    );
    let vwidth = _mm256_set1_epi32(width as i32);
    for y_win in y_winstart..y_winend {
        let y_abs = y as i32 + y_win;
        assert!(y_abs > 0);
        assert!((y_abs as u32) < height - 1);
        let y_winf = _mm256_set1_ps(y_win as f32);
        let mut pix_idx = pix_idx_rowstart;
        for x_win in (x_winstart..x_winend).step_by(8) {
            let x_abs = x as i32 + x_win;
            assert!(x_abs > 0);
            let x_abs = x_abs as u32;
            assert!(x_abs < width - 1);
            let mask = if (x_abs + 8) as i32 > x_winend + x as i32 {
                masks[(8 - (x_winend + x as i32 - x_abs as i32)) as usize]
            } else {
                mask_all
            };
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

            let dx = {
                let idx1 = _mm256_add_epi32(pix_idx, _mm256_set1_epi32(1));
                let idx2 = _mm256_sub_epi32(pix_idx, _mm256_set1_epi32(1));
                let v1 = _mm256_mask_i32gather_ps::<4>(_mm256_setzero_ps(), img_ptr, idx1, mask);
                let v2 = _mm256_mask_i32gather_ps::<4>(_mm256_setzero_ps(), img_ptr, idx2, mask);
                _mm256_sub_ps(v1, v2)
            };
            let dy = {
                let idx1 = _mm256_sub_epi32(pix_idx, vwidth);
                let idx2 = _mm256_add_epi32(pix_idx, vwidth);
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

            _mm256_store_si256(tmp_row_floor_p1.as_mut_ptr() as *mut __m256i, row_floor_p1);
            _mm256_store_si256(tmp_col_floor_p1.as_mut_ptr() as *mut __m256i, col_floor_p1);
            _mm256_store_si256(tmp_ori_floor.as_mut_ptr() as *mut __m256i, ori_floor);

            let msk = {
                let mut v = [0_i32; 8];
                _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, mask);
                v
            };
            for j in 0..8 {
                if msk[j] == 0 {
                    continue;
                }
                let ori_alt_ones = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
                let row_floor_p1 = _mm256_set1_epi32(tmp_row_floor_p1[j]);
                let col_floor_p1 = _mm256_set1_epi32(tmp_col_floor_p1[j]);
                let ori_single = _mm256_set1_epi32(tmp_ori_floor[j]);

                let ori_idx_offset =
                    _mm256_and_si256(_mm256_add_epi32(ori_single, ori_alt_ones), mod_nbins_mask);
                let idx_base = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_mullo_epi32(row_floor_p1, nhist_plus_two), n_bins),
                    _mm256_mullo_epi32(col_floor_p1, n_bins),
                );
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
            // TODO: make sure this is a shift
            pix_idx = _mm256_add_epi32(pix_idx, _mm256_set1_epi32(8));
        }
        pix_idx_rowstart = _mm256_add_epi32(pix_idx_rowstart, vwidth);
    }
    hist
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
unsafe fn normalize_hist_avx2(hist: &mut [f32], out: &mut [u8]) {
    use std::arch::x86_64::*;
    assert!(hist.len() % 8 == 0);

    // Single component cap computed from L2 norm
    let mut acc = _mm256_setzero_ps();
    for i in (0..hist.len()).step_by(8) {
        let val = _mm256_loadu_ps(&hist[i]);
        acc = _mm256_fmadd_ps(val, val, acc);
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps::<1>(acc);
    let red = _mm_add_ps(lo, hi);
    let red = _mm_hadd_ps(red, red);
    let red = _mm_hadd_ps(red, red);
    let l2norm = _mm_mul_ps(_mm_rsqrt_ss(red), red);
    let component_cap = _mm_mul_ps(
        _mm_broadcastss_ps(l2norm),
        _mm_set1_ps(DESCRIPTOR_MAGNITUDE_CAP),
    );
    let component_cap =
        _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128(component_cap)));
    // cap each component to component_cap
    for i in (0..hist.len()).step_by(8) {
        let val = _mm256_loadu_ps(&hist[i]);
        let gt_cap = _mm256_cmp_ps::<_CMP_GT_OQ>(val, component_cap);
        let capped = _mm256_blendv_ps(val, component_cap, gt_cap);
        _mm256_storeu_ps(hist.as_mut_ptr().add(i), capped);
    }
    // normalize l2 norm to DESCRIPTOR_L2_NORM
    let mut acc = _mm256_setzero_ps();
    for i in (0..hist.len()).step_by(8) {
        let val = _mm256_loadu_ps(&hist[i]);
        acc = _mm256_fmadd_ps(val, val, acc);
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps::<1>(acc);
    let red = _mm_add_ps(lo, hi);
    let red = _mm_hadd_ps(red, red);
    let red = _mm_hadd_ps(red, red);
    let rsqrt = _mm_rsqrt_ss(red);
    let rsqrt = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128(
        _mm_broadcastss_ps(rsqrt),
    )));
    let norm = _mm256_mul_ps(rsqrt, _mm256_set1_ps(DESCRIPTOR_L2_NORM));
    // saturating cast to u8
    for i in (0..hist.len()).step_by(8) {
        let val = _mm256_loadu_ps(&hist[i]);
        let valnorm = _mm256_mul_ps(val, norm);
        let valnorm = _mm256_cvtps_epi32(_mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(valnorm));
        let gt_255 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(valnorm, _mm256_set1_epi32(255)));
        let sat = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(valnorm),
            _mm256_castsi256_ps(_mm256_set1_epi32(255)),
            gt_255,
        ));
        // pack lowest byte of each f32 in sat into 8 bytes
        let shuf = _mm256_set_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 12, 8, 4, 0,
        );

        let shuffled = _mm256_shuffle_epi8(sat, shuf);
        let packed =
            _mm256_permutevar8x32_epi32(shuffled, _mm256_set_epi32(-1, -1, -1, -1, -1, -1, 4, 0));

        let store_mask = _mm256_set_epi64x(0, 0, 0, i64::MIN);
        _mm256_maskstore_epi64(out.as_mut_ptr().add(i) as *mut i64, store_mask, packed);
    }
}
