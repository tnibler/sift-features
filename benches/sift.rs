use divan::{black_box, Bencher};
use image::{ImageBuffer, Luma};
use nshare::IntoNdarray2 as _;

fn main() {
    divan::main();
}

type LumaFImage = ImageBuffer<Luma<f32>, Vec<f32>>;
struct OpenCVProcessing;

impl OpenCVProcessing {
    fn opencv_resize(img: &LumaFImage, width: u32, height: u32, method: i32) -> LumaFImage {
        use nshare::AsNdarray2;
        use opencv::prelude::*;

        let img = img.as_ndarray2();
        let mat = Mat::new_rows_cols_with_data(
            img.shape()[0] as i32,
            img.shape()[1] as i32,
            img.as_slice().unwrap(),
        )
        .unwrap();
        let mut res = Mat::default();
        opencv::imgproc::resize(
            &mat,
            &mut res,
            opencv::core::Size::new(width as i32, height as i32),
            0.,
            0.,
            method,
        )
        .unwrap();
        LumaFImage::from_vec(
            res.cols() as u32, // TODO: remove cast
            res.rows() as u32,
            res.data_typed().unwrap().to_vec(),
        )
        .unwrap()
    }
}

impl sift_features::Processing for OpenCVProcessing {
    fn gaussian_blur(img: &LumaFImage, sigma: f64) -> LumaFImage {
        use opencv::prelude::*;
        let img = img.clone().into_ndarray2();
        let mat = Mat::new_rows_cols_with_data(
            img.shape()[0] as i32,
            img.shape()[1] as i32,
            img.as_slice().unwrap(),
        )
        .unwrap();
        let mut res_mat = Mat::default();
        opencv::imgproc::gaussian_blur_def(
            &mat,
            &mut res_mat,
            opencv::core::Size::default(),
            sigma,
        )
        .unwrap();
        let result = LumaFImage::from_vec(
            res_mat.cols() as u32,
            res_mat.rows() as u32,
            res_mat.data_typed().unwrap().to_vec(),
        )
        .unwrap();
        result
    }

    fn resize_linear(img: &LumaFImage, width: u32, height: u32) -> LumaFImage {
        Self::opencv_resize(img, width, height, opencv::imgproc::INTER_LINEAR)
    }

    fn resize_nearest(img: &LumaFImage, width: u32, height: u32) -> LumaFImage {
        Self::opencv_resize(img, width, height, opencv::imgproc::INTER_NEAREST)
    }
}

const PATH: &str = "images/bird.jpg";

fn load_image() -> image::GrayImage {
    match image::open(PATH).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => panic!("wrong image type"),
    }
}

#[divan::bench]
fn sift_with_opencv_preprocess(bencher: Bencher) {
    let img = load_image();

    bencher.bench_local(|| {
        black_box(sift_features::sift_with_processing::<OpenCVProcessing>(
            &img, None,
        ))
    });
}

#[divan::bench]
fn opencv_sift(bencher: Bencher) {
    use opencv::features2d::Feature2DTrait;
    use opencv::prelude::*;
    let img = opencv::imgcodecs::imread(PATH, opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();

    bencher.bench_local(|| {
        let mut cvsift = opencv::features2d::SIFT::create_def().unwrap();
        let mut cvkps = opencv::core::Vector::new();
        let mut cvdescs = Mat::default();
        cvsift
            .detect_and_compute_def(&img, &opencv::core::no_array(), &mut cvkps, &mut cvdescs)
            .unwrap();
    });
}

#[divan::bench]
fn sift_no_preprocess(bencher: Bencher) {
    let img = load_image();
    let images = sift_features::precompute_images::<OpenCVProcessing>(&img);

    bencher.bench_local(|| black_box(sift_features::sift_with_precomputed(&images, None)));
}
