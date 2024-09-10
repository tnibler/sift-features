//! Run sift and do nothing.

use sift_features::SiftResult;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path1 = &args[1];
    let img1 = match image::open(path1).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => {
            eprintln!("wrong image type");
            return;
        }
    };

    let SiftResult {
        keypoints,
        descriptors: _,
    } = sift_features::sift(&img1, None);
    println!("{} keypoints", keypoints.len());
}
