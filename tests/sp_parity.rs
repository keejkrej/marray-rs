use marray::sp;
use ndarray::{array, Array1, Array2};
use std::f64::consts::PI;

#[test]
fn test_sp_linalg_parity() {
    let a = array![[4.0, 1.0, -1.0], [2.0, 5.0, -2.0], [1.0, 1.0, 2.0]];

    // Matrix inversion and determinant
    let inv = sp::linalg::inv(&a).unwrap();
    let id = a.dot(&inv);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((id[[i, j]] - expected).abs() < 1e-9);
        }
    }
    let det = sp::linalg::det(&a);
    assert!((det - 45.0).abs() < 1e-6);

    // LU decomposition
    let (p, l, u) = sp::linalg::lu(&a).unwrap();
    let plu = p.dot(&l).dot(&u);
    assert!((&plu - &a).iter().all(|&diff| diff.abs() < 1e-6));

    // QR decomposition
    let (q, r) = sp::linalg::qr(&a).unwrap();
    let qr = q.dot(&r);
    assert!((&qr - &a).iter().all(|&diff| diff.abs() < 1e-6));
    let q_t_q = q.t().dot(&q);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((q_t_q[[i, j]] - expected).abs() < 1e-6);
        }
    }

    // SVD
    let svd = sp::linalg::svd(&a);
    assert_eq!(svd.s.len(), 3);
    assert!(svd.s[0] >= svd.s[1] && svd.s[1] >= svd.s[2]);

    // Matrix Exponential
    let zero_mat = Array2::<f64>::zeros((2, 2));
    let exp_zero = sp::linalg::expm(&zero_mat).unwrap();
    assert!((exp_zero[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((exp_zero[[1, 1]] - 1.0).abs() < 1e-6);

    // Kronecker product and Block Diagonal
    let id2 = Array2::<f64>::eye(2);
    let kr = sp::linalg::kron(&id2, &id2);
    assert_eq!(kr.shape(), &[4, 4]);

    let bd = sp::linalg::block_diag(&[id2.clone(), id2.clone()]);
    assert_eq!(bd.shape(), &[4, 4]);
}

#[test]
fn test_sp_special_parity() {
    // Gamma and log-gamma
    assert!((sp::special::gamma(5.0) - 24.0).abs() < 1e-9);
    assert!((sp::special::gamma(0.5) - PI.sqrt()).abs() < 1e-9);
    assert!((sp::special::gammaln(5.0) - 24.0_f64.ln()).abs() < 1e-9);

    // Beta
    assert!((sp::special::beta(2.0, 3.0) - (1.0 / 12.0)).abs() < 1e-6);

    // Digamma
    assert!((sp::special::digamma(1.0) - (-0.5772156649)).abs() < 1e-5);

    // Error function and inverse
    assert!((sp::special::erf(0.0)).abs() < 1e-9);
    assert!((sp::special::erfinv(0.0)).abs() < 1e-9);
    let erf_val = sp::special::erf(0.5);
    assert!((sp::special::erfinv(erf_val) - 0.5).abs() < 1e-6);

    // Normal distribution CDF/quantiles
    assert!((sp::special::ndtr(0.0) - 0.5).abs() < 1e-9);
    assert!((sp::special::ndtri(0.5) - 0.0).abs() < 1e-9);

    // Logistic and logit
    assert!((sp::special::expit(0.0) - 0.5).abs() < 1e-9);
    assert!((sp::special::logit(0.5) - 0.0).abs() < 1e-9);

    // LogSumExp and Softmax
    let vec = [1.0, 2.0, 3.0];
    let lse = sp::special::logsumexp(&vec);
    let sm = sp::special::softmax(&vec);
    assert!((sm.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!((sm[2] - (3.0_f64 - lse).exp()).abs() < 1e-9);

    // Combinatorics
    assert_eq!(sp::special::factorial(5), 120.0);
    assert_eq!(sp::special::comb(5, 2), 10.0);
    assert_eq!(sp::special::perm(5, 2), 20.0);
}

#[test]
fn test_sp_stats_parity() {
    use sp::stats::ContinuousDistribution;

    // Norm distribution
    let norm = sp::stats::NormDist::new(0.0, 1.0);
    assert!((norm.cdf(0.0) - 0.5).abs() < 1e-7);
    assert!((norm.ppf(0.5) - 0.0).abs() < 1e-7);
    assert!((norm.pdf(0.0) - 1.0 / (2.0 * PI).sqrt()).abs() < 1e-7);

    // T distribution
    let t = sp::stats::TDist::new(10.0);
    assert!((t.cdf(0.0) - 0.5).abs() < 1e-4);

    // Chi2 distribution
    let chi = sp::stats::Chi2Dist::new(4.0);
    assert!((chi.mean() - 4.0).abs() < 1e-6);

    // Hypothesis tests: 1-sample t-test
    let sample = [1.0, 2.0, 3.0, 4.0, 5.0];
    let (t_stat, p_val) = sp::stats::ttest_1samp(&sample, 3.0);
    assert!(t_stat.abs() < 1e-9);
    assert!((p_val - 1.0).abs() < 1e-6);

    // Chi2 contingency test
    let obs = array![[10.0, 20.0], [20.0, 40.0]];
    let (chi2, p, dof, _) = sp::stats::chi2_contingency(&obs).unwrap();
    assert!(chi2 < 1e-6);
    assert_eq!(dof, 1);
    assert!((p - 1.0).abs() < 1e-4);

    // Descriptive stats
    let (cnt, min_val, max_val, mean, var, sk, kurt) = sp::stats::describe(&sample);
    assert_eq!(cnt, 5);
    assert_eq!(min_val, 1.0);
    assert_eq!(max_val, 5.0);
    assert_eq!(mean, 3.0);
    assert_eq!(var, 2.5);
    assert!(sk.abs() < 1e-6);
    assert!((kurt - (-1.2)).abs() < 0.2);

    // Wasserstein distance
    let u_vals = array![0.0, 1.0];
    let v_vals = array![1.0, 2.0];
    assert!((sp::stats::wasserstein_distance(&u_vals, &v_vals) - 1.0).abs() < 1e-6);
}

#[test]
fn test_sp_optimize_parity() {
    // 1D root finding (brentq)
    let root = sp::optimize::brentq(|x| x * x - 4.0, 0.0, 3.0, 1e-8, 1e-8, 100).unwrap();
    assert!((root - 2.0).abs() < 1e-6);

    // Newton root finding
    let newton_root = sp::optimize::newton(|x| x * x - 9.0, Some(|x| 2.0 * x), 2.0, 1e-8, 100).unwrap();
    assert!((newton_root - 3.0).abs() < 1e-6);

    // Nelder-Mead multivariate minimization: Rosenbrock function f(x, y) = (1-x)^2 + 100(y-x^2)^2
    let rosenbrock = |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
    let x0 = array![0.0, 0.0];
    let res = sp::optimize::minimize(rosenbrock, &x0, "Nelder-Mead", Some(1e-4), Some(500)).unwrap();
    assert!((res.x[0] - 1.0).abs() < 0.05);
    assert!((res.x[1] - 1.0).abs() < 0.05);

    // BFGS multivariate minimization on a simple parabola
    let parabola = |x: &Array1<f64>| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
    let res_bfgs = sp::optimize::minimize(parabola, &x0, "BFGS", Some(1e-6), Some(100)).unwrap();
    assert!((res_bfgs.x[0] - 2.0).abs() < 1e-4);
    assert!((res_bfgs.x[1] - 3.0).abs() < 1e-4);
}

#[test]
fn test_sp_integrate_parity() {
    // Quad integration of sin(x) from 0 to PI
    let (val, err) = sp::integrate::quad(|x| x.sin(), 0.0, PI, None, None).unwrap();
    assert!((val - 2.0).abs() < 1e-6);
    assert!(err < 1e-6);

    // Trapz and Simpson
    let x = array![0.0, 0.5 * PI, PI];
    let y = array![0.0, 1.0, 0.0];
    let simps_val = sp::integrate::simpson(&y, Some(&x), None);
    assert!((simps_val - 2.0).abs() < 0.15);

    // solve_ivp: dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
    let ode = |_t: f64, y: &Array1<f64>| -y;
    let y0 = array![1.0];
    let sol = sp::integrate::solve_ivp(ode, (0.0, 1.0), &y0, None, None, None);
    assert!((sol.y.last().unwrap()[0] - (-1.0_f64).exp()).abs() < 1e-3);
}

#[test]
fn test_sp_signal_parity() {
    // Filter design
    let (b, a, _gain) = sp::signal::butter(2, 0.2, "low");
    assert_eq!(b.len(), 3);
    assert_eq!(a.len(), 3);

    // Convolution
    let s1 = array![1.0, 2.0, 3.0];
    let s2 = array![0.0, 1.0, 0.5];
    let conv = sp::signal::convolve(&s1, &s2, sp::signal::ConvolveMode::Same);
    assert_eq!(conv.len(), 3);

    // Welch spectrum
    let signal_data: Vec<f64> = (0..256).map(|i| (2.0 * PI * 0.1 * (i as f64)).sin()).collect();
    let (freqs, psd) = sp::signal::welch(&signal_data, 64, 32);
    assert!(!freqs.is_empty());
    assert_eq!(freqs.len(), psd.len());

    // Savitzky-Golay filter
    let smoothed = sp::signal::savgol_filter(&signal_data, 5, 2);
    assert_eq!(smoothed.len(), 256);
}

#[test]
fn test_sp_ndimage_and_spatial_parity() {
    // ndimage Gaussian filter
    let img = Array2::<f64>::zeros((10, 10));
    let filtered = sp::ndimage::gaussian_filter(&img, 1.0);
    assert_eq!(filtered.dim(), (10, 10));

    // ndimage Labeling
    let mut binary = Array2::<bool>::from_elem((5, 5), false);
    binary[[1, 1]] = true;
    binary[[1, 2]] = true;
    binary[[3, 3]] = true;
    let (labeled, num_features) = sp::ndimage::label(&binary);
    assert_eq!(num_features, 2);
    assert_eq!(labeled[[1, 1]], labeled[[1, 2]]);
    assert_ne!(labeled[[1, 1]], labeled[[3, 3]]);

    // KDTree nearest neighbor
    let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let tree = sp::spatial::KDTree::new(&pts);
    let (dist, idx) = tree.query(&[0.1, 0.1], 1);
    assert_eq!(idx[0], 0);
    assert!(dist[0] < 0.2);

    // Convex Hull
    let hull = sp::spatial::ConvexHull::new(&pts).unwrap();
    assert_eq!(hull.vertices.len(), 4);
}

#[test]
fn test_sp_sparse_and_fft_parity() {
    // Sparse matrix CSR and solve
    let mut coo = sp::sparse::COOMatrix::empty((3, 3));
    coo.add_entry(0, 0, 5.0);
    coo.add_entry(1, 1, 3.0);
    coo.add_entry(2, 2, 2.0);
    let csr = coo.to_csr();
    let b = array![10.0, 6.0, 4.0];
    let x = sp::sparse::linalg::spsolve(&csr, &b).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-6);
    assert!((x[1] - 2.0).abs() < 1e-6);
    assert!((x[2] - 2.0).abs() < 1e-6);

    // FFT & IFFT roundtrip
    let real_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let rfft = sp::fft::rfft(&real_data);
    let roundtrip = sp::fft::irfft(&rfft, 8);
    for i in 0..8 {
        assert!((real_data[i] - roundtrip[i]).abs() < 1e-6);
    }

    let freqs = sp::fft::rfftfreq(8, 1.0);
    assert_eq!(freqs.len(), 5);
    assert_eq!(freqs[0], 0.0);
    assert_eq!(freqs[4], 0.5);
}
