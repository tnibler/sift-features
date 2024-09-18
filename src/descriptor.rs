use aligned_vec::{AVec, ConstAlign};
use itertools::{izip, Itertools as _};
use ndarray::{s, Array3, ArrayView2};

use crate::{DESCRIPTOR_N_BINS, DESCRIPTOR_N_HISTOGRAMS, LAMBDA_DESCR};

pub fn compute_descriptor(
    img: &ArrayView2<f32>,
    x: f32,
    y: f32,
    scale: f32,
    orientation: f32,
) -> impl IntoIterator<Item = u8> {
    let n_hist = DESCRIPTOR_N_HISTOGRAMS;
    let n_bins = DESCRIPTOR_N_BINS;
    let height = img.shape()[0];
    let width = img.shape()[1];
    let x = x.round() as usize;
    let y = y.round() as usize;
    const BIN_ANGLE_STEP: f32 = DESCRIPTOR_N_BINS as f32 / 360.0;
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
    let normalized_orienations: AVec<_, ConstAlign<ALIGN>> = AVec::from_iter(
        ALIGN,
        gradients_x
            .into_iter()
            .zip(gradients_y.iter())
            .map(|(x, y)| {
                let x: f64 = *x as f64;
                let y: f64 = *y as f64;
                ((y.atan2(x).to_degrees() + 360.0) % 360.0) as f32 - orientation
            }),
    );
    // Gradient magnitudes
    let magnitude: AVec<_, ConstAlign<ALIGN>> = AVec::from_iter(
        ALIGN,
        gradients_x
            .iter()
            .zip(gradients_y.iter())
            .map(|(x, y)| (x * x + y * y).sqrt()),
    );

    // Instead of 4*4 histograms, we work with 5*5 here so that the interpolation works out simpler
    // at the borders (surely possible to do differently as well).
    // The outermost histograms will be discarded.
    let mut hist: Array3<f32> = Array3::zeros((n_hist + 2, n_hist + 2, DESCRIPTOR_N_BINS));

    // Spread each sample point's contribution to its 8 neighbouring histograms based on its distance
    // from the histogram window's center and weighted by the sample's gradient magnitude.
    izip!(
        row_bins.iter(),
        col_bins.iter(),
        normalized_orienations.iter(),
        magnitude.iter(),
        weights.iter()
    )
    .for_each(|(row_bin, col_bin, orientation, mag, weight)| {
        // Subtracting 0.5 here because the trilinear interpolation (the reverse actually)
        // below works on the {-0.5, 0.5}^3 cube, but our histograms are located in {0, 1}^ cubes.
        let row_bin = row_bin - 0.5;
        let col_bin = col_bin - 0.5;
        let mag = mag * weight;
        let obin = orientation * BIN_ANGLE_STEP;
        let row_floor = row_bin.floor();
        let col_floor = col_bin.floor();
        let ori_floor = obin.floor();
        let row_frac = row_bin - row_floor;
        let col_frac = col_bin - col_floor;
        let ori_frac = obin - ori_floor;

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

    #[allow(clippy::reversed_empty_ranges)]
    let mut hist_flat = hist.slice_move(s![1..-1, 1..-1, ..]).into_flat();
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
    hist_flat.into_iter().map(move |x| {
        let x = (x * l2_normalizer).round() as i32;
        if x > (u8::MAX as i32) {
            u8::MAX
        } else {
            x as u8
        }
    })
}
