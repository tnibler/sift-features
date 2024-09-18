// This implementation of SIFT is derived from works by Rob Hess and Willow Garage Inc.
// It is made available under the terms of the MIT license included in the root of this repository.
//
// Copyright 2006-2010 Rob Hess
// Copyright 2009 Willow Garage Inc.
// Copyright 2024 Thomas Nibler

//! This crate contains an implemenation of the SIFT image descriptor.
//! It aims to be compatible with the implementation found in OpenCV's `feature2d` module
//! and you should be able to match features extracted with OpenCV and this crate.
//!
//! Useful resources:
//! - [1]: [Lowe 1999](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)
//! - [2]: [Lowe 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
//! - [3]: [Mikolajczyk 2004](https://robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_ijcv2004.pdf)
//! - [4]: [Rey-Otero 2014](https://www.ipol.im/pub/art/2014/82/article.pdf)
//!
//! The code tries to follow [4] (Anatomy of the SIFT Method) in particular.
//! It deviates in a few places to be compatible with the SIFT implementation OpenCV,
//! namely how histograms are smoothed, angle computations and some details in how the final
//! descriptor vector is calculated.

use std::cmp::min;
use std::f32::consts::PI as PI32;

use image::buffer::ConvertBuffer;
use image::imageops::{resize, FilterType};
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;
use itertools::{izip, Itertools};
use ndarray::{prelude::*, s, Array2, Array3, Axis};
use nshare::AsNdarray2;

mod descriptor;
mod local_extrema;

use local_extrema::local_extrema;

#[doc(hidden)]
pub use descriptor::compute_descriptor;

#[cfg(any(feature = "opencv", test))]
mod opencv_processing;
#[cfg(any(feature = "opencv", test))]
pub use opencv;
#[cfg(any(feature = "opencv", test))]
pub use opencv_processing::*;

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct SiftResult {
    pub keypoints: Vec<KeyPoint>,
    /// Array of shape `(keypoints.len(), 128)` containing the SIFT feature vectors in the same
    /// order as `keypoints`.
    pub descriptors: Array2<u8>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct SiftKeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: usize,
    pub scale: usize,
}

/// Extract SIFT features using default blur and resize implementations.
pub fn sift(img: &GrayImage, features_limit: Option<usize>) -> SiftResult {
    sift_with_processing::<ImageprocProcessing>(img, features_limit)
}

/// Extract SIFT features using provided blur and resize implementations.
pub fn sift_with_processing<P: Processing>(
    img: &GrayImage,
    features_limit: Option<usize>,
) -> SiftResult {
    sift_with_precomputed(&precompute_images::<P>(img), features_limit)
}

/// Basic image operations used by SIFT.
/// For testing or benchmarking, it's useful to use exactly the same blur and interpolation
/// procedures to obtain identical results and performance.
pub trait Processing {
    fn gaussian_blur(img: &LumaFImage, sigma: f64) -> LumaFImage;
    fn resize_linear(img: &LumaFImage, width: u32, height: u32) -> LumaFImage;
    fn resize_nearest(img: &LumaFImage, width: u32, height: u32) -> LumaFImage;
}

const SCALES_PER_OCTAVE: usize = 3;
const CONTRAST_THRESHOLD: f32 = 0.04;
const EDGE_THRESHOLD: f32 = 10.0;

/// λori in [4], radius around a keypoint considered for the gradient orientation histogram.
const ORIENTATION_HISTOGRAM_RADIUS: f32 = 1.5;
/// 3λori rounded up. For points closer to the image bounds than this, no gradient orientation
/// histogram can be compuited.
const IMAGE_BORDER: i32 = 5;

const ORIENTATION_HISTOGRAM_BINS: usize = 36;
/// λ_ori in Eq. (19) in [4]
const LAMBDA_ORI: f32 = 1.5;
const LAMBDA_DESCR: f32 = 3.0;

// See Section 4.2 in [4]
const DESCRIPTOR_N_HISTOGRAMS: usize = 4;
// See Section 4.2 in [4]
const DESCRIPTOR_N_BINS: usize = 8;
pub const DESCRIPTOR_SIZE: usize =
    DESCRIPTOR_N_HISTOGRAMS * DESCRIPTOR_N_HISTOGRAMS * DESCRIPTOR_N_BINS;

type LumaFImage = ImageBuffer<Luma<f32>, Vec<f32>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct ScaleSpacePoint {
    pub scale: usize,
    pub x: usize,
    pub y: usize,
}

#[doc(hidden)]
pub struct PrecomputedImages {
    pub scale_space: Vec<Array3<f32>>,
    pub dog: Vec<Array3<f32>>,
    pub n_octaves: usize,
}

#[doc(hidden)]
pub fn precompute_images<P: Processing>(img: &GrayImage) -> PrecomputedImages {
    let seed = create_seed_image::<P>(img);
    let min_axis = min(seed.width(), seed.height());
    let n_octaves: usize = ((min_axis as f32).log2() - 2.0).round() as usize + 1;

    let scale_space = build_gaussian_scale_space::<P>(seed, n_octaves);
    let dog = build_dog(n_octaves, &scale_space);
    PrecomputedImages {
        n_octaves,
        scale_space,
        dog,
    }
}

// Image preprocessing pulled out for easier benchmarking
#[doc(hidden)]
pub fn sift_with_precomputed(
    PrecomputedImages {
        scale_space,
        dog,
        n_octaves,
    }: &PrecomputedImages,
    features_limit: Option<usize>,
) -> SiftResult {
    let mut keypoints: Vec<SiftKeyPoint> = find_keypoints(*n_octaves, scale_space, dog).collect();
    if let Some(limit) = features_limit {
        if limit < keypoints.len() {
            keypoints.sort_unstable_by(|kp1, kp2| kp2.response.total_cmp(&kp1.response));
            keypoints.truncate(limit);
        }
    }
    let desc = compute_descriptors(scale_space, &keypoints);
    SiftResult {
        keypoints: keypoints
            .into_iter()
            .map(|kp| KeyPoint {
                // Undo initial upsampling by 2x for seed image
                x: kp.x * DELTA_MIN,
                y: kp.y * DELTA_MIN,
                size: kp.size * DELTA_MIN,
                angle: kp.angle,
                response: kp.response,
            })
            .collect(),
        descriptors: desc,
    }
}

/// Assumed blur level of input image
/// See Section 2.2 in [4].
const SIGMA_IN: f64 = 0.5;

/// Blur level of the seed image.
/// See Section 2.2 in [4].
const SIGMA_MIN: f64 = 0.8;

/// Inverse of the subsampling factor of first image in the gaussian scale space.
/// See Section 2.2 in [4].
const INV_DELTA_MIN: u32 = 2;

/// Subsampling factor of first image in the gaussian scale space.
/// See Section 2.2 in [4].
const DELTA_MIN: f32 = 0.5;

/// Compute upsampled and blurred seed image as described in [4], Eq. (6).
fn create_seed_image<P: Processing>(img: &GrayImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    // float image with pixel values [0; 1];
    let img_f32: LumaFImage = img.convert();
    // Initial upsampling step by DELTA_MIN
    // OpenCV doesn't use their normal resize function but instead `warpAffine` to upscale 2x. Why?
    let img_2x = P::resize_linear(
        &img_f32,
        img_f32.width() * INV_DELTA_MIN,
        img_f32.height() * INV_DELTA_MIN,
    );

    let sigma = (SIGMA_MIN * SIGMA_MIN - SIGMA_IN * SIGMA_IN).sqrt() * INV_DELTA_MIN as f64;

    P::gaussian_blur(&img_2x, sigma)
}

/// See Section 2.2 in [4]
fn build_gaussian_scale_space<P: Processing>(
    seed_img: LumaFImage,
    n_octaves: usize,
) -> Vec<Array3<f32>> {
    // Geometric series of blur sigmas within an octave as given in Eq. (7).
    // Each octave contains 3 additional images 0, SCALES_PER_OCTAVE+1, SCALES_PER_OCTAVE+2,
    // hence the +3 here and everywhere else.
    let m: f64 = 2_f64.powf(2.0 / SCALES_PER_OCTAVE as f64);
    let sigmas: Vec<f64> = (0..(SCALES_PER_OCTAVE as i32 + 3))
        .map(|s| {
            // right term under square root
            let a = m.powi(s - 1);
            // left term under square root
            let b = a * m;
            (b - a).sqrt() * SIGMA_MIN * INV_DELTA_MIN as f64
        })
        .collect();
    let create_octave = |initial: LumaFImage| {
        let mut imgs = Vec::with_capacity(SCALES_PER_OCTAVE + 3);
        imgs.push(initial);
        sigmas.iter().skip(1).for_each(|sigma| {
            let prev = imgs.last().unwrap();
            imgs.push(P::gaussian_blur(prev, *sigma));
        });
        imgs
    };
    let mut scale_space: Vec<Vec<LumaFImage>> = Vec::with_capacity(n_octaves);
    scale_space.push(create_octave(seed_img));
    for _ in 1..n_octaves {
        // The first image of each octave is the last (ignoring the two additional posterior ones)
        // image of the previous octave subsampled by a factor of 2.
        // See Eq. (8) in [4].
        let last_octave = &scale_space.last().unwrap();
        let initial = &last_octave[last_octave.len() - 3];
        let scaled_half = P::resize_nearest(initial, initial.width() / 2, initial.height() / 2);
        scale_space.push(create_octave(scaled_half));
    }
    assert_eq!(scale_space.len(), n_octaves);
    scale_space
        .iter()
        .map(|octave| {
            assert!(octave
                .windows(2)
                .all(|w| w[0].width() == w[1].width() && w[0].height() == w[1].height()));
            assert_eq!(octave.len(), SCALES_PER_OCTAVE + 3);
            let width = octave[0].width() as usize;
            let height = octave[0].height() as usize;
            let mut mat: Array3<f32> = Array3::zeros((SCALES_PER_OCTAVE + 3, height, width));
            octave.iter().enumerate().for_each(|(i, img)| {
                mat.slice_mut(s![i, .., ..]).assign(&img.as_ndarray2());
            });
            mat
        })
        .collect()
}

/// Difference of Gaussians, woof.
/// See Section 3.1 in [4]
fn build_dog(n_octaves: usize, scale_space: &[Array3<f32>]) -> Vec<Array3<f32>> {
    assert!(scale_space.len() == n_octaves);
    let dog: Vec<Array3<f32>> = scale_space
        .iter()
        .map(|octave| &octave.slice(s![1.., .., ..]) - &octave.slice(s![..-1, .., ..]))
        .collect();
    assert!(dog.iter().all(|d| d.shape()[0] == SCALES_PER_OCTAVE + 2));
    dog
}

fn find_keypoints<'a>(
    n_octaves: usize,
    scale_space: &'a [Array3<f32>],
    dogs: &'a [Array3<f32>],
) -> impl Iterator<Item = SiftKeyPoint> + 'a {
    assert!(scale_space.len() == n_octaves);
    (0..n_octaves).flat_map(move |octave| {
        let dog = &dogs[octave];
        assert!(dog.shape()[0] == SCALES_PER_OCTAVE + 2);
        (1..=SCALES_PER_OCTAVE).flat_map(move |scale_in_octave| {
            find_extrema_in_dog_img(dog, scale_space, octave, scale_in_octave)
        })
    })
}

/// t in Section 4.1.C
const ORIENTATION_HISTOGRAM_LOCALMAX_RATIO: f32 = 0.8;

fn find_extrema_in_dog_img<'a>(
    dog: &'a Array3<f32>,
    scale_space: &'a [Array3<f32>],
    octave: usize,
    scale_in_octave: usize,
) -> Box<dyn Iterator<Item = SiftKeyPoint> + 'a> {
    assert!(dog.shape()[0] == SCALES_PER_OCTAVE + 2);
    assert!(scale_in_octave > 0);
    assert!(scale_in_octave < dog.shape()[0] - 1);
    let dogslice = dog.slice(s![scale_in_octave - 1..scale_in_octave + 2, .., ..]);
    assert!(dogslice.shape()[0] == 3);
    let curr = dogslice.index_axis(Axis(0), 1);

    let height = curr.nrows() as i32;
    let width = curr.ncols() as i32;

    if height < 2 * IMAGE_BORDER || width < 2 * IMAGE_BORDER {
        return Box::new(std::iter::empty());
    }

    let extremum_threshold: f32 = (0.5 * CONTRAST_THRESHOLD / SCALES_PER_OCTAVE as f32).floor();
    let extrema: Vec<_> = local_extrema(&dogslice, IMAGE_BORDER as usize, extremum_threshold);

    let result_iter = extrema.into_iter().flat_map(move |(x_initial, y_initial)| {
        let dogslice = dog.slice(s![scale_in_octave - 1..scale_in_octave + 2, .., ..]);
        assert!(dogslice.shape()[0] == 3);
        let InterpolateResult {
            offset_scale,
            offset_x,
            offset_y,
            point,
        } = match interpolate_extremum(
            dog.into(),
            ScaleSpacePoint {
                scale: scale_in_octave,
                x: x_initial,
                y: y_initial,
            },
        ) {
            Some(r) => r,
            None => return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = _>>,
        };

        let dogslice = dog.slice(s![(point.scale - 1)..(point.scale + 2), .., ..]);
        let curr = dogslice.index_axis(Axis(0), 1);
        // discard low contrast extrema
        let contrast =
            extremum_contrast(dogslice, point.x, point.y, offset_scale, offset_x, offset_y).abs();

        if contrast * SCALES_PER_OCTAVE as f32 <= CONTRAST_THRESHOLD {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = _>>;
        }

        // Discard extrema located on edges
        if extremum_is_on_edge(curr, point) {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = _>>;
        }

        let octave_scale_factor = 2_f32.powi(octave as i32);

        // Called sigma in [4]
        let kp_scale = SIGMA_MIN as f32
            * 2_f32.powf((point.scale as f32 + offset_scale) / SCALES_PER_OCTAVE as f32)
            * 2.;

        let kp_x = (point.x as f32 + offset_x) * octave_scale_factor;
        let kp_y = (point.y as f32 + offset_y) * octave_scale_factor;
        // Side length of patch over which gradient orientation histogram is computed.
        // See Eq. (19) in [4]
        let radius: i32 = (3. * ORIENTATION_HISTOGRAM_RADIUS * kp_scale).round() as i32;
        let hist = gradient_direction_histogram(
            scale_space[octave].slice(s![point.scale, .., ..]),
            point.x as u32,
            point.y as u32,
            radius,
            LAMBDA_ORI * kp_scale,
            ORIENTATION_HISTOGRAM_BINS,
        );
        let histogram_max = hist
            .iter()
            .copied()
            .max_by(f32::total_cmp)
            .expect("vec is not empty");
        let localmax_threshold = histogram_max * ORIENTATION_HISTOGRAM_LOCALMAX_RATIO;

        // Extract keypoint reference orientations, Section 4.1.C in [4]
        let kps_with_ref_orientation = (0..hist.len()).filter_map(move |k| {
            // h_k- and h_k+ in [4].
            // Histogram indices wrap around since bins correspond to angles.
            let k_minus = if k > 0 { k - 1 } else { hist.len() - 1 };
            let k_plus = if k < hist.len() - 1 { k + 1 } else { 0 };
            let is_local_max = hist[k] > hist[k_minus] && hist[k] > hist[k_plus];
            let is_close_to_global_max = hist[k] >= localmax_threshold;
            if is_local_max && is_close_to_global_max {
                // argmax of the quadratic function interpolating h_k-, h_k, h_k+
                // See Eq. (23) in [4]
                let interp =
                    (hist[k_minus] - hist[k_plus]) / (hist[k_minus] - 2.0 * hist[k] + hist[k_plus]);
                let bin: f32 = k as f32 + 0.5 * interp;
                let bin = if bin < 0.0 {
                    hist.len() as f32 + bin
                } else if bin >= hist.len() as f32 {
                    bin - hist.len() as f32
                } else {
                    bin
                };
                // The angles are shuffled around to match OpenCV
                let kp_angle: f32 = 360.0 - (360.0 / hist.len() as f32) * bin;
                Some(SiftKeyPoint {
                    x: kp_x,
                    y: kp_y,
                    size: kp_scale * octave_scale_factor,
                    response: contrast,
                    octave,
                    scale: point.scale,
                    angle: kp_angle,
                })
            } else {
                None
            }
        });
        Box::new(kps_with_ref_orientation)
    });
    Box::new(result_iter)
}

#[derive(Copy, Clone)]
struct InterpolateResult {
    pub point: ScaleSpacePoint,
    pub offset_scale: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

const MAX_INTERPOLATION_STEPS: usize = 5;

/// Scale space extrema are initially identified on the grid of discrete pixels in a particular
/// image in the scale space. The real DoG function approximated by the stack of DoG images is continous
/// though, and the actual extremum may not fall exactly on a sampling point (scale, row, column).
/// To get a better approximation of the extremum's location, the second order Taylor expansion is
/// used to fit the DoG function around a point and the local extremum of this quadratic is used as
/// a keypoint.
/// See P18-19 in [4].
fn interpolate_extremum(
    dog: ArrayView3<f32>,
    ScaleSpacePoint {
        mut scale,
        mut x,
        mut y,
    }: ScaleSpacePoint,
) -> Option<InterpolateResult> {
    assert!(dog.shape()[0] == SCALES_PER_OCTAVE + 2);
    for _ in 0..MAX_INTERPOLATION_STEPS {
        let prev = &dog.slice(s![scale - 1, .., ..]);
        let curr = &dog.slice(s![scale, .., ..]);
        let next = &dog.slice(s![scale + 1, .., ..]);

        // 3D Gradient
        let g1 = (next[(y, x)] - prev[(y, x)]) / 2.;
        let g2 = (curr[(y + 1, x)] - curr[(y - 1, x)]) / 2.;
        let g3 = (curr[(y, x + 1)] - curr[(y, x - 1)]) / 2.;

        // Hessian matrix
        let value2x = curr[(y, x)] * 2.;
        let h11 = next[(y, x)] + prev[(y, x)] - value2x;
        let h12 = (next[(y + 1, x)] - next[(y - 1, x)] - prev[(y + 1, x)] + prev[(y - 1, x)]) / 4.;
        let h13 = (next[(y, x + 1)] - next[(y, x - 1)] - prev[(y, x + 1)] + prev[(y, x - 1)]) / 4.;
        let h22 = curr[(y + 1, x)] + curr[(y - 1, x)] - value2x;
        let h33 = curr[(y, x + 1)] + curr[(y, x - 1)] - value2x;
        let h23 = (curr[(y + 1, x + 1)] - curr[(y + 1, x - 1)] - curr[(y - 1, x + 1)]
            + curr[(y - 1, x - 1)])
            / 4.;

        // Solve for α* as shown in Eq. (14) by inverting the hessian
        let det = h11 * h22 * h33 - h11 * h23 * h23 - h12 * h12 * h33 + 2. * h12 * h13 * h23
            - h13 * h13 * h22;
        let hinv11 = (h22 * h33 - h23 * h23) / det;
        let hinv12 = (h13 * h23 - h12 * h33) / det;
        let hinv13 = (h12 * h23 - h13 * h22) / det;
        let hinv22 = (h11 * h33 - h13 * h13) / det;
        let hinv23 = (h12 * h13 - h11 * h23) / det;
        let hinv33 = (h11 * h22 - h12 * h12) / det;

        // dot product of gradient vector with hessian inverse.
        // Solution vector α* is (offset_scale, offset_row, offset_col)
        let offset_scale = -(hinv11 * g1 + hinv12 * g2 + hinv13 * g3);
        let offset_x = -(hinv13 * g1 + hinv23 * g2 + hinv33 * g3);
        let offset_y = -(hinv12 * g1 + hinv22 * g2 + hinv23 * g3);

        // If offsets are outside the interval [-0.5; 0.5] the extremum belongs
        // to a different pixel or scale and should be rejected here.
        let valid_interval = 0.5;
        if offset_scale.abs() < valid_interval
            && offset_x.abs() < valid_interval
            && offset_y.abs() < valid_interval
        {
            // extremum of quadratic function is valid
            return Some(InterpolateResult {
                offset_scale,
                offset_y,
                offset_x,
                point: ScaleSpacePoint { scale, y, x },
            });
        }
        // Interpolation step rejected, update discrete extremum coordinates
        // and retry interpolation.
        x = (x as isize + offset_x.round() as isize) as usize;
        y = (y as isize + offset_y.round() as isize) as usize;
        scale = (scale as isize + offset_scale.round() as isize) as usize;

        if !(1..=SCALES_PER_OCTAVE).contains(&scale)
            || x < IMAGE_BORDER as usize
            || x >= curr.shape()[1] - IMAGE_BORDER as usize
            || y < IMAGE_BORDER as usize
            || y >= curr.shape()[0] - IMAGE_BORDER as usize
        {
            return None;
        }
    }
    // did not converge with in iteration limit
    None
}

/// Based on P11, Eq. (2) and (3) in [2]. This step is not mentioned in [4].
fn extremum_contrast(
    dogslice: ArrayView3<f32>,
    x: usize,
    y: usize,
    interp_offset_scale: f32,
    interp_offset_x: f32,
    interp_offset_y: f32,
) -> f32 {
    assert!(dogslice.shape()[0] == 3);
    let prev = dogslice.index_axis(Axis(0), 0);
    let curr = dogslice.index_axis(Axis(0), 1);
    let next = dogslice.index_axis(Axis(0), 2);
    // 3D Gradient
    let g1 = (next[(y, x)] - prev[(y, x)]) / 2.;
    let g2 = (curr[(y + 1, x)] - curr[(y - 1, x)]) / 2.;
    let g3 = (curr[(y, x + 1)] - curr[(y, x - 1)]) / 2.;
    // Value of the interpolating function at x̂ in [2], or α* in [4].
    let interp = interp_offset_scale * g1 + interp_offset_y * g2 + interp_offset_x * g3;
    curr[(y, x)] + interp / 2.
    // extremum_contrast.abs() * SCALES_PER_OCTAVE as f32 > CONTRAST_THRESHOLD
}

/// Measures "edgeness" of a point the ratio between eigenvalues of the Hessian matrix.
/// P382, Eq. (17) and Eq. (18) in [4]
fn extremum_is_on_edge(
    dog_curr: ArrayView2<f32>,
    ScaleSpacePoint { scale: _, y, x }: ScaleSpacePoint,
) -> bool {
    assert!(x > 0 && x < dog_curr.shape()[1] - 1);
    assert!(y > 0 && y < dog_curr.shape()[0] - 1);
    let val2x = dog_curr[(y, x)] * 2.0;
    let h11 = dog_curr[(y + 1, x)] + dog_curr[(y - 1, x)] - val2x;
    let d22 = dog_curr[(y, x + 1)] + dog_curr[(y, x - 1)] - val2x;

    let h12 = (dog_curr[(y + 1, x + 1)] - dog_curr[(y + 1, x - 1)] - dog_curr[(y - 1, x + 1)]
        + dog_curr[(y - 1, x - 1)])
        / 4.;

    let tr = d22 + h11;
    let det = d22 * h11 - h12 * h12;
    if det <= 0. {
        return true;
    }
    // edgeness = tr^2 / det
    //     edgeness > (C_edge + 1)^2 / C_edge
    // <=> tr^2 * C_edge > (C_edge + 1)^2 * det
    (tr * tr * EDGE_THRESHOLD) > (EDGE_THRESHOLD + 1.0).powi(2) * det
}

/// Histogram of gradient directions in square patch of side length 2*radius around (row, col).
/// See Section 4.1 in [4].
fn gradient_direction_histogram(
    img: ArrayView2<f32>,
    x: u32,
    y: u32,
    radius: i32,
    sigma: f32,
    n_bins: usize,
) -> Vec<f32> {
    assert!(n_bins >= 2);
    // Denominator of exponent in Eq. (20) in [4], used to compute weights
    let grad_weight_scale = -1.0 / (2.0 * sigma * sigma);

    // x/y gradients are weighted by their distance from the point at (row, col).
    // weights holds the exponent of the weighting factor in Eq. (20) in [4]
    let (grads_x, grads_y, grad_weights): (Vec<f32>, Vec<f32>, Vec<f32>) = (-radius..=radius)
        .filter_map(|y_patch| {
            if y_patch <= -(y as i32) {
                return None;
            }
            let y: i64 = i64::from(y) + i64::from(y_patch);
            if y <= 0 || y as usize >= img.shape()[0] - 1 {
                return None;
            }
            Some((y as usize, y_patch))
        })
        .flat_map(|(y_img, y_patch)| {
            (-radius..=radius)
                .filter_map(|x_patch| {
                    if x_patch <= -(x as i32) {
                        return None;
                    }
                    let x = x as isize + x_patch as isize;
                    if x <= 0 || x as usize >= img.shape()[1] - 1 {
                        return None;
                    }
                    Some((x as usize, x_patch))
                })
                .map(move |(x_img, x_patch)| {
                    let dx = img[(y_img, x_img + 1)] - img[(y_img, x_img - 1)];
                    let dy = img[(y_img - 1, x_img)] - img[(y_img + 1, x_img)];
                    // squared euclidian distance from (row, col) * weighting factor
                    let w = (y_patch * y_patch + x_patch * x_patch) as f32 * grad_weight_scale;
                    (dx, dy, w)
                })
        })
        .multiunzip();

    // Finalizing the term in Eq. (20) in [4]
    let grad_weights = grad_weights.into_iter().map(|x| x.exp());
    // gradient magnitudes
    let magnitudes = grads_x
        .iter()
        .zip(&grads_y)
        .map(|(x, y)| (x * x + y * y).sqrt());
    let orientations = grads_x
        .iter()
        .copied()
        .zip(&grads_y)
        .map(|(x, y)| f64::from(*y).atan2(x.into()) as f32);

    // Range of angles (radians) assigned to one histogram bin
    let bin_angle_step = n_bins as f32 / (PI32 * 2.);
    // Histogram bin index as given by Eq. (21) in [4]
    let hist_bin = orientations.into_iter().map(|ori| {
        assert!((-PI32..=PI32).contains(&ori));
        let raw_bin = bin_angle_step * ori;
        // raw_bin is in range [-PI * (n_bins / 2PI); PI * (n_bins / 2PI)]
        //                    =[-n_bins / 2; n_bins / 2];
        assert!(-(n_bins as f32) / 2. <= raw_bin && raw_bin <= n_bins as f32 / 2.);
        let bin: i32 = raw_bin.round() as i32;
        if bin >= n_bins as i32 {
            (bin - n_bins as i32) as usize
        } else if bin < 0 {
            assert!(bin + n_bins as i32 >= 0);
            (bin + n_bins as i32) as usize
        } else {
            bin as usize
        }
    });

    // The gradient orientation histogram undergoes a final smoothing step.
    // In [4], smoothing is done by convolving 6 times with a kernel of [1/3, 1/3, 1/3].
    // In OpenCV, the kernel [1/16, 4/16, 6/16, 4/16, 1/16] is instead used one time only.
    // raw_hist has length n_bins + 4 because the convolution is circular/cyclic and  wraps around,
    // so we copy the first and last 2 values to the other end of the histogram to get this wrapping.
    let mut raw_hist = vec![0.0; n_bins + 4];
    izip!(hist_bin, magnitudes, grad_weights).for_each(|(bin, mag, weight)| {
        raw_hist[bin + 2] += weight * mag;
    });
    raw_hist[1] = raw_hist[n_bins + 1];
    raw_hist[0] = raw_hist[n_bins];
    raw_hist[n_bins + 2] = raw_hist[2];
    raw_hist[n_bins + 3] = raw_hist[3];
    let mut hist = vec![0.; n_bins];
    for i in 2..n_bins + 2 {
        hist[i - 2] = (raw_hist[i - 2] + raw_hist[i + 2]) * (1. / 16.)
            + (raw_hist[i - 1] + raw_hist[i + 1]) * (4. / 16.)
            + raw_hist[i] * 6. / 16.;
    }
    hist
}

fn compute_descriptors(scale_space: &[Array3<f32>], keypoints: &[SiftKeyPoint]) -> Array2<u8> {
    let mut desc = Array2::zeros((keypoints.len(), DESCRIPTOR_SIZE));
    desc.rows_mut()
        .into_iter()
        .zip(keypoints)
        .for_each(|(mut row, kp)| {
            let img = &scale_space[kp.octave].index_axis(Axis(0), kp.scale);
            let angle = 360.0 - kp.angle;
            // δ_o in Eq. (9) in [4] with δ_min = 2
            let octave_scale_factor = 2_f32.powi(-(kp.octave as i32));
            let kp_size = kp.size * octave_scale_factor;
            compute_descriptor(
                img,
                kp.x * octave_scale_factor,
                kp.y * octave_scale_factor,
                kp_size,
                angle,
                row.as_slice_mut().expect("is contiguous"),
            );
        });
    desc
}

/// Uses `imageproc` implementations of gaussian blur and resizing.
pub struct ImageprocProcessing;

impl Processing for ImageprocProcessing {
    fn gaussian_blur(img: &LumaFImage, sigma: f64) -> LumaFImage {
        gaussian_blur_f32(img, sigma as f32)
    }

    fn resize_linear(img: &LumaFImage, width: u32, height: u32) -> LumaFImage {
        resize(img, width, height, FilterType::Triangle)
    }

    fn resize_nearest(img: &LumaFImage, width: u32, height: u32) -> LumaFImage {
        resize(img, width, height, FilterType::Nearest)
    }
}
