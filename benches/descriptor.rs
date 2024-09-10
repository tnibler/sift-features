use divan::{black_box, Bencher};
use image::{buffer::ConvertBuffer, ImageBuffer, Luma};
use nshare::IntoNdarray2;

fn main() {
    divan::main();
}

const PATH: &str = "images/bird.jpg";

fn load_image() -> image::GrayImage {
    match image::open(PATH).unwrap().grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => panic!("wrong image type"),
    }
}

#[divan::bench(sample_count = 1000)]
fn sift_descriptor(bencher: Bencher) {
    let img: ImageBuffer<Luma<f32>, _> = load_image().convert();
    let img = img.into_ndarray2();

    bencher.bench_local(|| black_box(sift::compute_descriptor(&img.view(), 100., 100., 2.1, 123.)));
}
