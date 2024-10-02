use std::{path::PathBuf, str::FromStr};

use average::Estimate;
use image::{GrayImage, ImageBuffer, Luma};
use itertools::Itertools;
use opencv::{
    core::{
        no_array, DMatch, KeyPointTrait as _, KeyPointTraitConst, Mat, MatTraitConst,
        MatTraitConstManual as _, Size, Vector,
    },
    features2d,
    imgcodecs::{imread, ImreadModes},
    imgproc::{resize, INTER_LINEAR},
    prelude::{DescriptorMatcherTrait as _, Feature2DTrait},
};
use sift_features::{sift_with_processing, KeyPoint, OpenCVProcessing, SiftResult};

const SCALES: &[f32] = &[0.8, 0.6, 0.5, 0.4];

type LumaFImage = ImageBuffer<Luma<f32>, Vec<f32>>;

fn to_cv_keypoints(kp: &KeyPoint) -> opencv::core::KeyPoint {
    let mut cvkp = opencv::core::KeyPoint::default().unwrap();
    cvkp.set_pt(opencv::core::Point_ { x: kp.x, y: kp.y });
    cvkp.set_size(kp.size);
    cvkp.set_angle(kp.angle);
    cvkp.set_octave(1);
    cvkp.set_response(kp.response);
    cvkp
}

fn main() -> Result<(), ()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Required args: IMAGES");
        return Err(());
    }
    let filename_width = args
        .iter()
        .map(|s| {
            let pb = PathBuf::from_str(s)
                .unwrap()
                .file_name()
                .expect("paths must be files")
                .to_owned();
            pb.to_str().unwrap().to_string().len()
        })
        .max()
        .unwrap()
        + 4;

    let use_precise_upscale = false;
    let mut cv_sift = features2d::SIFT::create(0, 3, 0.04, 10., 1.6, use_precise_upscale).unwrap();
    println!(
        "{:<20} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}",
        "File", "Scl", "Avg", "Max", "q50", "q75", "q90"
    );
    for path in &args[1..] {
        let cv_img_orig = opencv::imgcodecs::imread(path, ImreadModes::IMREAD_GRAYSCALE.into())
            .unwrap_or_else(|_| panic!("failed to open image {}", path));
        assert_eq!(cv_img_orig.channels(), 1);
        assert_eq!(cv_img_orig.typ(), opencv::core::CV_8U);
        let path = PathBuf::from(path);
        let filename = path.file_name().unwrap().to_str().unwrap().to_owned();

        let img_orig = GrayImage::from_vec(
            cv_img_orig.cols() as u32,
            cv_img_orig.rows() as u32,
            cv_img_orig.data_typed().unwrap().to_vec(),
        )
        .unwrap();
        let mut cv_kp_orig = Vector::new();
        let mut cv_desc_orig = Mat::default();
        cv_sift
            .detect_and_compute_def(
                &cv_img_orig,
                &no_array(),
                &mut cv_kp_orig,
                &mut cv_desc_orig,
            )
            .unwrap();

        let SiftResult {
            keypoints: kp_orig,
            descriptors: desc_orig,
        } = sift_with_processing::<OpenCVProcessing>(&img_orig, None);
        for scale in SCALES {
            let mut cv_img_scaled = Mat::default();
            resize(
                &cv_img_orig,
                &mut cv_img_scaled,
                Size::default(),
                (*scale).into(),
                (*scale).into(),
                INTER_LINEAR,
            )
            .expect("failed to resize image");
            let mut cv_kp_scaled = Vector::new();
            let mut cv_desc_scaled = Mat::default();
            cv_sift
                .detect_and_compute_def(
                    &cv_img_scaled,
                    &no_array(),
                    &mut cv_kp_scaled,
                    &mut cv_desc_scaled,
                )
                .unwrap();

            let mut cv_matches = opencv::core::Vector::new();
            let mut matcher =
                opencv::features2d::BFMatcher::new(opencv::core::NORM_L2, true).unwrap();
            matcher.add(&cv_desc_orig).unwrap();
            matcher
                .match_(&cv_desc_scaled, &mut cv_matches, &opencv::core::no_array())
                .unwrap();

            let cv_dists = cv_matches
                .iter()
                .map(|mtch| {
                    let kp_orig = cv_kp_orig.get(mtch.train_idx as usize).unwrap();
                    let kp_scaled = cv_kp_scaled.get(mtch.query_idx as usize).unwrap();
                    let dx = kp_orig.pt().x - kp_scaled.pt().x / scale;
                    let dy = kp_orig.pt().y - kp_scaled.pt().y / scale;
                    (dx * dx + dy * dy).sqrt() as f64
                })
                .collect_vec();

            let img_scaled = GrayImage::from_vec(
                cv_img_scaled.cols() as u32,
                cv_img_scaled.rows() as u32,
                cv_img_scaled.data_typed().unwrap().to_vec(),
            )
            .unwrap();
            let SiftResult {
                keypoints: kp_scaled,
                descriptors: desc_scaled,
            } = sift_with_processing::<OpenCVProcessing>(&img_scaled, None);

            let mut matches = opencv::core::Vector::new();
            let mut matcher =
                opencv::features2d::BFMatcher::new(opencv::core::NORM_L2, true).unwrap();
            matcher
                .add(
                    &Mat::new_rows_cols_with_data(
                        desc_orig.shape()[0] as i32,
                        desc_orig.shape()[1] as i32,
                        desc_orig.as_slice().unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap();
            matcher
                .match_(
                    &Mat::new_rows_cols_with_data(
                        desc_scaled.shape()[0] as i32,
                        desc_scaled.shape()[1] as i32,
                        desc_scaled.as_slice().unwrap(),
                    )
                    .unwrap(),
                    &mut matches,
                    &opencv::core::no_array(),
                )
                .unwrap();
            let dists = matches
                .iter()
                .map(|mtch| {
                    let kp_orig = kp_orig.get(mtch.train_idx as usize).unwrap();
                    let kp_scaled = kp_scaled.get(mtch.query_idx as usize).unwrap();
                    let dx = kp_orig.x - kp_scaled.x / scale;
                    let dy = kp_orig.y - kp_scaled.y / scale;
                    (dx * dx + dy * dy).sqrt() as f64
                })
                .collect_vec();

            #[derive(Debug, Clone, Copy)]
            struct Stats {
                pub mean: f64,
                pub max: f64,
                pub q50: f64,
                pub q75: f64,
                pub q90: f64,
            }

            fn calc_stats(dists: &[f64]) -> Stats {
                let mean: average::Mean = dists.iter().collect();
                let mean = mean.mean();
                let max = *dists.iter().max_by(|a, b| f64::total_cmp(a, b)).unwrap();
                let mut q90 = average::Quantile::new(0.9);
                let mut q75 = average::Quantile::new(0.75);
                let mut q50 = average::Quantile::new(0.5);
                dists.iter().copied().for_each(|d| {
                    q90.add(d);
                    q75.add(d);
                    q50.add(d);
                });
                Stats {
                    mean,
                    max,
                    q50: q50.quantile(),
                    q75: q75.quantile(),
                    q90: q90.quantile(),
                }
            };
            let cv_stats = calc_stats(&cv_dists);
            let stats = calc_stats(&dists);
            println!(
                "{filename:<20} {scale:<5} {:<5.01} {:<5.01} {:<5.01} {:<5.01} {:<5.01}",
                cv_stats.mean / stats.mean,
                cv_stats.max / stats.max,
                cv_stats.q50 / stats.q50,
                cv_stats.q75 / stats.q75,
                cv_stats.q90 / stats.q90,
            );
        }
    }
    Ok(())
}
