//! Run this crate's SIFT and OpenCV's on two images and output matches in matches-rust.jpg and
//! matches-opencv.jpg.

use sift_features::opencv;

use opencv::core::KeyPoint as CVKeyPoint;
use opencv::core::ToInputArray;
use opencv::core::Vector as CVVec;
use opencv::features2d::Feature2DTrait;
use opencv::prelude::*;

use sift_features::{KeyPoint, OpenCVProcessing, SiftResult};

fn to_cv_keypoints(kp: &KeyPoint) -> opencv::core::KeyPoint {
    let mut cvkp = opencv::core::KeyPoint::default().unwrap();
    cvkp.set_pt(opencv::core::Point_ { x: kp.x, y: kp.y });
    cvkp.set_size(kp.size);
    cvkp.set_angle(kp.angle);
    cvkp.set_octave(1);
    cvkp.set_response(kp.response);
    cvkp
}

fn write_matches_img(
    img1: Mat,
    kp1: CVVec<CVKeyPoint>,
    desc1: impl MatTraitConst + ToInputArray,
    img2: Mat,
    kp2: CVVec<CVKeyPoint>,
    desc2: impl MatTraitConst + ToInputArray,
    name: &str,
) {
    let mut matcher = opencv::features2d::BFMatcher::new(opencv::core::NORM_L2, true).unwrap();
    matcher.add(&desc1).unwrap();
    let mut matches = opencv::core::Vector::new();
    matcher
        .match_(&desc2, &mut matches, &opencv::core::no_array())
        .unwrap();
    let mut out_img = Mat::default();
    opencv::features2d::draw_matches_def(&img2, &kp2, &img1, &kp1, &matches, &mut out_img).unwrap();
    opencv::imgcodecs::imwrite_def(name, &out_img).unwrap();
}

fn main() -> Result<(), ()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Required args: IMAGE1 IMAGE2");
        return Err(());
    }
    let path1 = &args[1];
    let path2 = &args[2];
    let img1 = match image::open(path1).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => {
            eprintln!("wrong image type");
            return Err(());
        }
    };
    let img2 = match image::open(path2).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => {
            eprintln!("wrong image type");
            return Err(());
        }
    };
    let cvimg1 = opencv::imgcodecs::imread(path1, opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();
    let cvimg2 = opencv::imgcodecs::imread(path2, opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();

    let mut cvsift = opencv::features2d::SIFT::create_def().unwrap();
    let mut cvkp1 = opencv::core::Vector::new();
    let mut cvdesc1 = Mat::default();
    cvsift
        .detect_and_compute_def(&cvimg1, &opencv::core::no_array(), &mut cvkp1, &mut cvdesc1)
        .unwrap();
    let mut cvdesc2 = Mat::default();
    let mut cvkp2 = opencv::core::Vector::new();
    cvsift
        .detect_and_compute_def(&cvimg2, &opencv::core::no_array(), &mut cvkp2, &mut cvdesc2)
        .unwrap();
    write_matches_img(
        cvimg1.clone(),
        cvkp1.clone(),
        cvdesc1.clone(),
        cvimg2.clone(),
        cvkp2,
        cvdesc2,
        "matches-opencv.jpg",
    );

    let SiftResult {
        keypoints: my_kp1,
        descriptors: my_desc1,
    } = sift_features::sift_with_processing::<OpenCVProcessing>(&img1, None);

    let SiftResult {
        keypoints: my_kp2,
        descriptors: my_desc2,
    } = sift_features::sift_with_processing::<OpenCVProcessing>(&img2, None);

    write_matches_img(
        cvimg1,
        my_kp1.into_iter().map(|kp| to_cv_keypoints(&kp)).collect(),
        Mat::new_rows_cols_with_data(
            my_desc1.shape()[0] as i32,
            my_desc1.shape()[1] as i32,
            my_desc1.as_slice().unwrap(),
        )
        .unwrap(),
        cvimg2,
        my_kp2.into_iter().map(|kp| to_cv_keypoints(&kp)).collect(),
        Mat::new_rows_cols_with_data(
            my_desc2.shape()[0] as i32,
            my_desc2.shape()[1] as i32,
            my_desc2.as_slice().unwrap(),
        )
        .unwrap(),
        "matches-rust.jpg",
    );
    Ok(())
}
