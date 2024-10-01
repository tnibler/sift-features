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
    if args.len() != 2 {
        eprintln!("Required args: IMAGE1");
        return Err(());
    }
    let path1 = &args[1];
    let cvimg1 = opencv::imgcodecs::imread(path1, opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();

    let mut cvsift = opencv::features2d::SIFT::create(0, 3, 0.04, 10., 1.6, true).unwrap();
    let mut cvkp1 = opencv::core::Vector::new();
    let mut cvdesc1 = Mat::default();
    cvsift
        .detect_and_compute_def(&cvimg1, &opencv::core::no_array(), &mut cvkp1, &mut cvdesc1)
        .unwrap();
    println!("{} keypoints", cvkp1.len());
    Ok(())
}
