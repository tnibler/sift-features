use crate::{LumaFImage, Processing};

pub struct OpenCVProcessing;

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
        //if method == opencv::imgproc::INTER_NEAREST {
        opencv::imgproc::resize(
            &mat,
            &mut res,
            opencv::core::Size::new(width as i32, height as i32),
            0.,
            0.,
            method,
        )
        .unwrap();
        //} else {
        //    opencv::imgproc::warp_affine(
        //        &mat,
        //        &mut res,
        //        &opencv::core::Mat::from_slice_2d(&[[0.5, 0., 0.], [0., 0.5, 0.]]).unwrap(),
        //        opencv::core::Size::new(width as i32, height as i32),
        //        opencv::imgproc::WARP_INVERSE_MAP,
        //        opencv::core::BorderTypes::BORDER_TRANSPARENT.into(),
        //        opencv::core::VecN([0.; 4]),
        //    )
        //    .unwrap();
        //}
        LumaFImage::from_vec(
            res.cols() as u32,
            res.rows() as u32,
            res.data_typed().unwrap().to_vec(),
        )
        .unwrap()
    }
}

impl Processing for OpenCVProcessing {
    fn gaussian_blur(img: &LumaFImage, sigma: f64) -> LumaFImage {
        use nshare::IntoNdarray2;
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
