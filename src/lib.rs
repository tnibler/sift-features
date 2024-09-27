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

#[cfg(any(test, feature = "opencv"))]
mod opencv_processing;
#[cfg(any(test, feature = "opencv"))]
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
const DESCRIPTOR_SIZE: usize =
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
    let y_end = height - IMAGE_BORDER;
    let y_begin = IMAGE_BORDER;

    let x_end = width - IMAGE_BORDER;
    let x_begin = IMAGE_BORDER;
    assert!(x_end >= IMAGE_BORDER);
    let extrema: Vec<_> = (y_begin..y_end)
        .flat_map(move |y_initial| {
            (x_begin..x_end)
                .filter(move |x_initial| {
                    point_is_local_extremum(dogslice, *x_initial as usize, y_initial as usize)
                })
                .map(move |x_initial| (x_initial, y_initial))
        })
        .collect();

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
                x: x_initial as usize,
                y: y_initial as usize,
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

fn point_is_local_extremum(dogslice: ArrayView3<f32>, x: usize, y: usize) -> bool {
    #[inline(always)]
    fn values_around(arr: &ArrayView2<f32>, y: usize, x: usize) -> impl Iterator<Item = f32> {
        [
            arr[(y - 1, x - 1)],
            arr[(y - 1, x)],
            arr[(y - 1, x + 1)],
            arr[(y, x - 1)],
            arr[(y, x + 1)],
            arr[(y + 1, x - 1)],
            arr[(y + 1, x)],
            arr[(y + 1, x + 1)],
        ]
        .into_iter()
    }

    assert!(dogslice.shape()[0] == 3);
    let prev = dogslice.index_axis(Axis(0), 0);
    let curr = dogslice.index_axis(Axis(0), 1);
    let next = dogslice.index_axis(Axis(0), 2);

    // Discard extrema with values below this threshold.
    // This is taken from OpenCV, but Section 3.3 in [4] uses different values.
    let threshold: f32 = (0.5 * CONTRAST_THRESHOLD / SCALES_PER_OCTAVE as f32).floor();

    assert!(x > 0 && y > 0 && x < curr.shape()[1] && y < curr.shape()[0]);

    let val = curr[(y, x)];
    if val.abs() <= threshold {
        return false;
    } else if val > 0.0 {
        if val
            >= values_around(&curr, y, x)
                .max_by(f32::total_cmp)
                .expect("sequence not empty")
            && val
                >= values_around(&prev, y, x)
                    .max_by(f32::total_cmp)
                    .expect("sequence not empty")
            && val
                >= values_around(&next, y, x)
                    .max_by(f32::total_cmp)
                    .expect("sequence not empty")
        {
            let _11p = prev[(y, x)];
            let _11n = next[(y, x)];
            return val >= _11p.max(_11n);
        }
    } else {
        debug_assert!(val < 0.0);
        if val
            <= values_around(&curr, y, x)
                .min_by(f32::total_cmp)
                .expect("sequence not empty")
            && val
                <= values_around(&prev, y, x)
                    .min_by(f32::total_cmp)
                    .expect("sequence not empty")
            && val
                <= values_around(&next, y, x)
                    .min_by(f32::total_cmp)
                    .expect("sequence not empty")
        {
            let _11p = prev[(y, x)];
            let _11n = next[(y, x)];
            return val <= _11p.min(_11n);
        }
    }
    false
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
        .for_each(|(row, kp)| {
            let img = &scale_space[kp.octave].index_axis(Axis(0), kp.scale);
            let angle = 360.0 - kp.angle;
            // δ_o in Eq. (9) in [4] with δ_min = 2
            let octave_scale_factor = 2_f32.powi(-(kp.octave as i32));
            let kp_size = kp.size * octave_scale_factor;
            let kpdesc = compute_descriptor(
                img,
                kp.x * octave_scale_factor,
                kp.y * octave_scale_factor,
                kp_size,
                angle,
            );
            row.into_iter()
                .zip(kpdesc)
                .for_each(|(el, descriptor_component)| *el = descriptor_component);
        });
    desc
}

#[doc(hidden)]
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

    // Instead of 4*4 histograms, we work with 5*5 here so that the interpolation works out simpler
    // at the borders (surely possible to do differently as well).
    // The outermost histograms will be discarded.
    let mut hist: Array3<f32> = Array3::zeros((n_hist + 2, n_hist + 2, DESCRIPTOR_N_BINS));

    let (gradients_x, gradients_y, row_bins, col_bins, weights): (
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
    ) = (-radius..=radius)
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
        .multiunzip();
    // Different weighting than in [4]
    let weight_scale = -2. / (n_hist.pow(2) as f32);
    let weights = weights
        .into_iter()
        .map(|x| (x * weight_scale).exp())
        .collect_vec();
    // Gradient orientations in patch normalized wrt to the keypoint's reference orientation.
    let normalized_orienations = gradients_x
        .iter()
        .zip(&gradients_y)
        .map(|(x, y)| {
            let x: f64 = *x as f64;
            let y: f64 = *y as f64;
            ((y.atan2(x).to_degrees() + 360.0) % 360.0) as f32 - orientation
        })
        .collect_vec();
    // Gradient magnitudes
    let magnitude = gradients_x
        .into_iter()
        .zip(&gradients_y)
        .map(|(x, y)| (x * x + y * y).sqrt())
        .collect_vec();

    // Spread each sample point's contribution to its 8 neighbouring histograms based on its distance
    // from the histogram window's center and weighted by the sample's gradient magnitude.
    izip!(
        row_bins,
        col_bins,
        normalized_orienations,
        magnitude,
        weights
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
