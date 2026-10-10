//! Numerical fidelity tests against Python reference fixtures.
//! Verifies exact floating point parity with NumPy, SciPy, scikit-learn, scikit-image, and numpy-ml.

use mlab::{np, numpy_ml, skimage, sklearn, sp};
use ndarray::{array, Array2, Array3};
use serde_json::Value;
use std::fs;
use std::path::Path;

fn load_fixtures() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/python_reference.json");
    let content = fs::read_to_string(&path).expect("Failed to read python_reference.json");
    serde_json::from_str(&content).expect("Failed to parse fixture JSON")
}

#[test]
fn test_fidelity_numpy() {
    let f = load_fixtures();
    let np_ref = &f["numpy"];

    // linspace
    let lin = np::linspace(0.0, 1.0, 5);
    let expected_lin: Vec<f64> = np_ref["linspace"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (a, b) in lin.iter().zip(expected_lin.iter()) {
        assert!((a - b).abs() < 1e-9);
    }

    // logspace
    let log = np::logspace(0.0, 3.0, 4, 10.0);
    let expected_log: Vec<f64> = np_ref["logspace"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (a, b) in log.iter().zip(expected_log.iter()) {
        assert!((a - b).abs() < 1e-9);
    }

    // geomspace
    let geom = np::geomspace(1.0, 1000.0, 4);
    let expected_geom: Vec<f64> = np_ref["geomspace"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (a, b) in geom.iter().zip(expected_geom.iter()) {
        assert!((a - b).abs() < 1e-6);
    }

    // cumsum & diff
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let cs = np::cumsum(&x);
    let expected_cs: Vec<f64> = np_ref["cumsum"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (a, b) in cs.iter().zip(expected_cs.iter()) {
        assert!((a - b).abs() < 1e-9);
    }

    // percentiles
    assert_eq!(np::percentile(&x, 25.0), np_ref["percentile_25"].as_f64().unwrap());
    assert_eq!(np::percentile(&x, 50.0), np_ref["percentile_50"].as_f64().unwrap());
    assert_eq!(np::percentile(&x, 75.0), np_ref["percentile_75"].as_f64().unwrap());

    // linalg
    let m = array![[1.0, 2.0], [3.0, 4.0]];
    let inv = np::linalg::inv(&m).unwrap();
    let expected_inv = np_ref["inv"].as_array().unwrap();
    for r in 0..2 {
        for c in 0..2 {
            let exp = expected_inv[r][c].as_f64().unwrap();
            assert!((inv[[r, c]] - exp).abs() < 1e-9);
        }
    }

    let det = np::linalg::det(&m);
    assert!((det - np_ref["det"].as_f64().unwrap()).abs() < 1e-9);

    let norm_val = np::linalg::rsnorm(&x, Some(2));
    assert!((norm_val - np_ref["norm_l2"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn test_fidelity_scipy() {
    let f = load_fixtures();
    let sp_ref = &f["scipy"];

    // special
    assert!((sp::special::gamma(5.0) - sp_ref["gamma_5"].as_f64().unwrap()).abs() < 1e-9);
    assert!((sp::special::gamma(0.5) - sp_ref["gamma_half"].as_f64().unwrap()).abs() < 1e-9);
    assert!((sp::special::gammaln(5.0) - sp_ref["gammaln_5"].as_f64().unwrap()).abs() < 1e-9);
    assert!((sp::special::beta(2.0, 3.0) - sp_ref["beta_2_3"].as_f64().unwrap()).abs() < 1e-9);
    assert!((sp::special::digamma(1.0) - sp_ref["digamma_1"].as_f64().unwrap()).abs() < 1e-5);
    assert!((sp::special::erf(0.5) - sp_ref["erf_half"].as_f64().unwrap()).abs() < 1e-6);
    assert!((sp::special::erfinv(0.5) - sp_ref["erfinv_half"].as_f64().unwrap()).abs() < 1e-6);
    assert!((sp::special::expit(0.0) - sp_ref["expit_0"].as_f64().unwrap()).abs() < 1e-9);
    assert!((sp::special::logit(0.5) - sp_ref["logit_half"].as_f64().unwrap()).abs() < 1e-9);
    assert!((sp::special::logsumexp(&[1.0, 2.0, 3.0]) - sp_ref["logsumexp"].as_f64().unwrap()).abs() < 1e-9);

    let sm = sp::special::softmax(&[1.0, 2.0, 3.0]);
    let exp_sm: Vec<f64> = sp_ref["softmax"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (a, b) in sm.iter().zip(exp_sm.iter()) {
        assert!((a - b).abs() < 1e-9);
    }

    assert!((sp::special::jv(0.0, 2.4048255577) - sp_ref["jv_0_24"].as_f64().unwrap()).abs() < 1e-6);
    assert!((sp::special::iv(0.0, 1.0) - sp_ref["iv_0_1"].as_f64().unwrap()).abs() < 1e-6);

    // stats
    use sp::stats::ContinuousDistribution;
    let norm = sp::stats::NormDist::new(0.0, 1.0);
    assert!((norm.cdf(0.0) - sp_ref["norm_cdf_0"].as_f64().unwrap()).abs() < 1e-7);
    assert!((norm.ppf(0.5) - sp_ref["norm_ppf_05"].as_f64().unwrap()).abs() < 1e-7);

    let t = sp::stats::TDist::new(10.0);
    assert!((t.cdf(0.0) - sp_ref["t_cdf_0"].as_f64().unwrap()).abs() < 1e-4);

    let (t_stat, p_val) = sp::stats::ttest_1samp(&[1.0, 2.0, 3.0, 4.0, 5.0], 3.0);
    assert!((t_stat - sp_ref["ttest_1samp_stat"].as_f64().unwrap()).abs() < 1e-9);
    assert!((p_val - sp_ref["ttest_1samp_pval"].as_f64().unwrap()).abs() < 1e-6);

    assert!((sp::stats::wasserstein_distance(&array![0.0, 1.0], &array![1.0, 2.0]) - sp_ref["wasserstein_distance"].as_f64().unwrap()).abs() < 1e-9);

    // signal, optimize, integrate
    let (b, a, _) = sp::signal::butter(2, 0.2, "low");
    let exp_b: Vec<f64> = sp_ref["butter_b"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let exp_a: Vec<f64> = sp_ref["butter_a"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (x, y) in b.iter().zip(exp_b.iter()) {
        assert!((x - y).abs() < 1e-6);
    }
    for (x, y) in a.iter().zip(exp_a.iter()) {
        assert!((x - y).abs() < 1e-6);
    }

    let root = sp::optimize::brentq(|x| x * x - 4.0, 0.0, 3.0, 1e-8, 1e-8, 100).unwrap();
    assert!((root - sp_ref["brentq_root"].as_f64().unwrap()).abs() < 1e-6);

    let (quad_val, _) = sp::integrate::quad(|x| x.sin(), 0.0, std::f64::consts::PI, None, None).unwrap();
    assert!((quad_val - sp_ref["quad_val"].as_f64().unwrap()).abs() < 1e-6);
}

#[test]
fn test_fidelity_sklearn() {
    let f = load_fixtures();
    let skl_ref = &f["sklearn"];

    // StandardScaler
    let x_skl = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]];
    let mut scaler = sklearn::preprocessing::StandardScaler::new();
    let x_scaled = scaler.fit_transform(&x_skl);
    let exp_trans = skl_ref["scaler_transform"].as_array().unwrap();
    for r in 0..3 {
        for c in 0..3 {
            let exp = exp_trans[r][c].as_f64().unwrap();
            assert!((x_scaled[[r, c]] - exp).abs() < 1e-6);
        }
    }

    // LinearRegression
    let x_reg = array![[1.0], [2.0], [3.0], [4.0]];
    let y_reg = array![2.0, 4.0, 6.0, 8.0];
    let mut lr = sklearn::linear_model::LinearRegression::new(true);
    lr.fit(&x_reg, &y_reg).unwrap();
    let lr_pred = lr.predict(&array![[5.0]]);
    assert!((lr_pred[0] - skl_ref["lr_pred"][0].as_f64().unwrap()).abs() < 1e-6);

    // Ridge
    let mut ridge = sklearn::linear_model::Ridge::new(1.0, true);
    ridge.fit(&x_reg, &y_reg).unwrap();
    let ridge_pred = ridge.predict(&array![[5.0]]);
    assert!((ridge_pred[0] - skl_ref["ridge_pred"][0].as_f64().unwrap()).abs() < 1e-6);

    // Metrics
    let y_true = array![0.0, 1.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.0, 1.0, 0.0, 0.0, 1.0];
    assert_eq!(sklearn::metrics::accuracy_score(&y_true, &y_pred), skl_ref["accuracy"].as_f64().unwrap());
    assert_eq!(sklearn::metrics::precision_score(y_true.as_slice().unwrap(), y_pred.as_slice().unwrap(), "binary"), skl_ref["precision"].as_f64().unwrap());
    assert!((sklearn::metrics::recall_score(y_true.as_slice().unwrap(), y_pred.as_slice().unwrap(), "binary") - skl_ref["recall"].as_f64().unwrap()).abs() < 1e-6);
    assert!((sklearn::metrics::f1_score(y_true.as_slice().unwrap(), y_pred.as_slice().unwrap(), "binary") - skl_ref["f1"].as_f64().unwrap()).abs() < 1e-6);
}

#[test]
fn test_fidelity_skimage() {
    let f = load_fixtures();
    let skimage_ref = &f["skimage"];

    // RGB to HSV
    let mut rgb_f = Array3::<f64>::zeros((1, 1, 3));
    rgb_f[[0, 0, 0]] = 1.0;
    let hsv = skimage::color::rgb2hsv(&rgb_f);
    let exp_hsv = skimage_ref["rgb2hsv_val"].as_array().unwrap();
    assert_eq!(hsv[[0, 0, 0]], exp_hsv[0].as_f64().unwrap());
    assert_eq!(hsv[[0, 0, 1]], exp_hsv[1].as_f64().unwrap());
    assert_eq!(hsv[[0, 0, 2]], exp_hsv[2].as_f64().unwrap());

    // Otsu thresholding
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
    assert_eq!(thresh, skimage_ref["otsu_thresh"].as_f64().unwrap());
}

#[test]
fn test_fidelity_numpy_ml() {
    let f = load_fixtures();
    let nml_ref = &f["numpy_ml"];

    let x_reg = array![[1.0], [2.0], [3.0], [4.0]];
    let y_reg = array![2.0, 4.0, 6.0, 8.0];
    let mut r_nml = numpy_ml::linear_models::RidgeRegression::new(1.0, true);
    r_nml.fit(&x_reg, &y_reg).unwrap();
    let pred = r_nml.predict(&array![[5.0]]).unwrap();
    assert!((pred[0] - nml_ref["ridge_pred"][0].as_f64().unwrap()).abs() < 1e-6);
}
