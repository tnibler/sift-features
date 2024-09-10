# SIFT features for Rust

This crate contains an implemenation of the SIFT image descriptor.
It aims to be compatible with the implementation found in OpenCV's `feature2d` module
and you should be able to match features extracted with OpenCV and this crate.

Useful resources:
- [1]: [Lowe 1999](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)
- [2]: [Lowe 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [3]: [Mikolajczyk 2004](https://robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_ijcv2004.pdf)
- [4]: [Rey-Otero 2014](https://www.ipol.im/pub/art/2014/82/article.pdf)

The code tries to follow [4] (Anatomy of the SIFT Method) in particular.
It deviates in a few places to be compatible with the SIFT implementation OpenCV,
namely how histograms are smoothed, angle computations and some details in how the final
descriptor vector is calculated.
