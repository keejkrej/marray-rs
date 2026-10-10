use mlab::skimage;
use ndarray::{Array2, Array3};

#[test]
fn test_skimage_color_parity() {
    let mut rgb = Array3::<u8>::zeros((4, 4, 3));
    // Red pixel
    rgb[[0, 0, 0]] = 255;
    rgb[[0, 0, 1]] = 0;
    rgb[[0, 0, 2]] = 0;

    let gray = skimage::color::rgb2gray(&rgb);
    assert_eq!(gray.dim(), (4, 4));
    // Red channel weight ~ 0.299 * 255 = 76
    assert_eq!(gray[[0, 0]], 76);

    let rgb_roundtrip = skimage::color::gray2rgb(&gray);
    assert_eq!(rgb_roundtrip.dim(), (4, 4, 3));
    assert_eq!(rgb_roundtrip[[0, 0, 0]], 76);
    assert_eq!(rgb_roundtrip[[0, 0, 1]], 76);
    assert_eq!(rgb_roundtrip[[0, 0, 2]], 76);

    // RGB to HSV
    let mut rgb_f = Array3::<f64>::zeros((1, 1, 3));
    rgb_f[[0, 0, 0]] = 1.0; // Red
    let hsv = skimage::color::rgb2hsv(&rgb_f);
    assert_eq!(hsv[[0, 0, 0]], 0.0); // Hue = 0
    assert_eq!(hsv[[0, 0, 1]], 1.0); // Sat = 1
    assert_eq!(hsv[[0, 0, 2]], 1.0); // Val = 1
}

#[test]
fn test_skimage_filters_parity() {
    let mut img = Array2::<u8>::zeros((16, 16));
    for i in 4..12 {
        for j in 4..12 {
            img[[i, j]] = 255;
        }
    }

    // Gaussian filter
    let smoothed = skimage::filters::gaussian(&img, 1.0);
    assert_eq!(smoothed.dim(), (16, 16));
    assert!(smoothed[[8, 8]] > 100);

    // Edge filters: Sobel, Prewitt, Laplace
    let edges_sobel = skimage::filters::sobel(&img);
    assert_eq!(edges_sobel.dim(), (16, 16));
    assert!(edges_sobel[[4, 8]] > 0);

    let img_f = img.mapv(|v| v as f64);
    let edges_prewitt = skimage::filters::prewitt(&img_f);
    assert_eq!(edges_prewitt.dim(), (16, 16));

    let laplace = skimage::filters::laplace(&img_f);
    assert_eq!(laplace.dim(), (16, 16));

    // Otsu thresholding on bimodal image
    let mut bimodal = Array2::<u8>::zeros((16, 16));
    for i in 0..8 {
        for j in 0..16 {
            bimodal[[i, j]] = 50;
        }
    }
    for i in 8..16 {
        for j in 0..16 {
            bimodal[[i, j]] = 200;
        }
    }
    let thresh = skimage::filters::threshold_otsu(&bimodal);
    assert_eq!(thresh, 50.0);
}

#[test]
fn test_skimage_morphology_and_measure_parity() {
    let mut binary = Array2::<bool>::from_elem((10, 10), false);
    for i in 2..8 {
        for j in 2..8 {
            binary[[i, j]] = true;
        }
    }

    // Disk and Square structuring elements
    let d = skimage::morphology::disk(1);
    assert_eq!(d.dim(), (3, 3));

    let sq = skimage::morphology::square(3);
    assert_eq!(sq.dim(), (3, 3));

    // Opening and Closing
    let opened = skimage::morphology::binary_opening(&binary, &sq);
    assert_eq!(opened.dim(), (10, 10));

    let closed = skimage::morphology::binary_closing(&binary, &sq);
    assert_eq!(closed.dim(), (10, 10));

    // Connected component labeling
    let (labels, num_features) = skimage::measure::label(&binary);
    assert_eq!(num_features, 1);
    assert_eq!(labels[[5, 5]], 1);
    assert_eq!(labels[[0, 0]], 0);

    // Regionprops
    let props = skimage::measure::regionprops(&labels, num_features);
    assert_eq!(props.len(), 1);
    assert_eq!(props[0].area, 36);

    // Perimeter
    let perim = skimage::measure::perimeter(&binary);
    assert!(perim > 0.0);
}

#[test]
fn test_skimage_exposure_and_metrics_parity() {
    let mut img = Array2::<f64>::zeros((8, 8));
    for i in 0..8 {
        for j in 0..8 {
            img[[i, j]] = (i * 8 + j) as f64 / 64.0;
        }
    }

    // Rescale intensity
    let rescaled = skimage::exposure::rescale_intensity(&img, (0.0, 1.0), (0.0, 255.0));
    assert_eq!(rescaled[[0, 0]], 0.0);
    assert!((rescaled[[7, 7]] - 251.015).abs() < 1.0);

    // Metrics: MSE, PSNR, SSIM
    let img_copy = img.clone();
    let mse = skimage::metrics::mean_squared_error(&img, &img_copy);
    assert_eq!(mse, 0.0);

    let psnr_val = skimage::metrics::peak_signal_noise_ratio(&img, &img_copy, 1.0);
    assert!(psnr_val.is_infinite());

    let ssim_val = skimage::metrics::structural_similarity(&img, &img_copy, 3);
    assert!((ssim_val - 1.0).abs() < 1e-6);
}
