//! Usage:
//! ./snapshot write image.jpg snapshot.jpg
//! ./snapshot test image.jpg snapshot.jpg

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::exit,
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use sift_features::{sift_with_processing, KeyPoint, OpenCVProcessing, SiftResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Snapshot {
    pub keypoints: Vec<KeyPoint>,
    pub descriptors: Vec<Vec<u8>>,
}

fn write_snapshot(
    SiftResult {
        keypoints,
        descriptors,
    }: SiftResult,
    out_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let descriptors = descriptors
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect_vec();
    fs::write(
        out_path,
        serde_json::to_string(&Snapshot {
            keypoints,
            descriptors,
        })?,
    )?;
    Ok(())
}

fn test_snapshot(
    SiftResult {
        keypoints,
        descriptors,
    }: SiftResult,
    snapshot_path: &Path,
) -> Result<bool, Box<dyn Error>> {
    let snap: Snapshot = serde_json::from_str(&fs::read_to_string(snapshot_path)?)?;
    let mut pass = true;
    if keypoints.len() != snap.keypoints.len() {
        pass = false;
        eprintln!(
            "Number of keypoints mismatch:\nExpected:\t{}\nActual:\t{}",
            snap.keypoints.len(),
            keypoints.len()
        );
    }
    let descriptors = descriptors
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect_vec();
    if descriptors.len() != snap.descriptors.len() {
        pass = false;
        eprintln!(
            "Number of descriptors mismatch:\nExpected:\t{}\nActual:\t{}",
            snap.descriptors.len(),
            descriptors.len()
        );
    }
    for (i, (snap_kp, kp)) in snap.keypoints.iter().zip(keypoints.iter()).enumerate() {
        const COORD_TOL: f32 = 1e-5;
        const ANGLE_TOL: f32 = 1e-2;
        let x_match = (snap_kp.x - kp.x).abs() <= COORD_TOL;
        let y_match = (snap_kp.y - kp.y).abs() <= COORD_TOL;
        let size_match = (snap_kp.size - kp.size).abs() <= COORD_TOL;
        let angle_match = (snap_kp.angle - kp.angle).abs() <= ANGLE_TOL;
        let response_match = (snap_kp.response - kp.response).abs() <= COORD_TOL;
        let all_match = x_match && y_match && size_match && angle_match && response_match;
        let none_match = !(x_match || y_match || size_match || angle_match || response_match);
        if none_match {
            pass = false;
            eprintln!("All keypoint fields mismatch");
            eprintln!("Expected: {:?}", snap_kp);
            eprintln!("Actual  : {:?}", kp);
            break;
        }
        if !all_match {
            pass = false;
            eprintln!("Keypoint {i} mismatch");
            eprintln!("Expected: {:?}", snap_kp);
            eprintln!("Actual  : {:?}", kp);
        }
    }
    if !pass {
        return Ok(pass);
    }
    for (i, (snap_desc, desc)) in snap.descriptors.iter().zip(&descriptors).enumerate() {
        const MAX_DIFF: i32 = 2;
        const MAX_DIFF_COUNT: usize = 12;
        let (max_diff_idx, max_diff) = snap_desc
            .iter()
            .zip(desc.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .enumerate()
            .max_by_key(|(_i, d)| *d)
            .expect("vec not empty");
        if max_diff > MAX_DIFF {
            eprintln!("Descriptor {i} mismatch: diff={max_diff} at index {max_diff_idx}\nExpected: {snap_desc:?}\nActual  : {desc:?}");
            eprintln!("{:?}", keypoints[i]);
            eprintln!("{:?}", snap.keypoints[i]);
            pass = false;
            break;
        }
        let diff_count = snap_desc
            .iter()
            .zip(desc.iter())
            .filter(|(a, b)| *a != *b)
            .count();
        if diff_count > MAX_DIFF_COUNT {
            eprintln!("Descriptor {i} mismatch in {diff_count} places:\nExpected: {snap_desc:?}\nActual  : {desc:?}");
            eprintln!("{:?}", keypoints[i]);
            eprintln!("{:?}", snap.keypoints[i]);
            pass = false;
            break;
        }
    }
    Ok(pass)
}

fn main() {
    let args = std::env::args().collect_vec();
    if args.len() != 4 || !(args[1] == "write" || args[1] == "test") {
        eprintln!(
            r#"
Usage:
./snapshot write [IMAGE] [SNAPSHOT]
./snapshot test [IMAGE] [SNAPSHOT]
        "#
        );
        exit(1);
    }
    let what = &args[1];
    let image_path = PathBuf::from(&args[2]);
    let snapshot_path = PathBuf::from(&args[3]);

    let image = match image::open(&image_path) {
        Ok(img) => match img.grayscale() {
            image::DynamicImage::ImageLuma8(img) => img,
            _ => {
                eprintln!("Wrong/unsupported image type");
                exit(1);
            }
        },
        Err(err) => {
            eprintln!("Error opening image '{}': {:?}", image_path.display(), err);
            exit(1);
        }
    };
    let sift_result = sift_with_processing::<OpenCVProcessing>(&image, None);
    match what.as_str() {
        "write" => {
            let res = write_snapshot(sift_result, &snapshot_path);
            if let Err(err) = res {
                eprintln!("Error: {:?}", err);
                exit(1);
            }
        }
        "test" => match test_snapshot(sift_result, &snapshot_path) {
            Err(err) => {
                eprintln!("Error: {:?}", err);
                exit(1);
            }
            Ok(false) => {
                eprintln!("FAIL");
                exit(1);
            }
            Ok(true) => {
                eprintln!("PASS");
            }
        },
        _ => unreachable!(),
    };
}
