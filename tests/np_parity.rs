use mlab_rs::np;
use ndarray::{array, Array1, Array2};

#[test]
fn test_np_array_creation_and_shaping() {
    // zeros, ones, full, eye
    let z: Array2<f64> = np::zeros((3, 4));
    assert_eq!(z.shape(), &[3, 4]);
    assert!(z.iter().all(|&v| v == 0.0));

    let o: Array1<f64> = np::ones(5);
    assert_eq!(o.shape(), &[5]);
    assert!(o.iter().all(|&v| v == 1.0));

    let f = np::full((2, 3), 42.0);
    assert_eq!(f.shape(), &[2, 3]);
    assert!(f.iter().all(|&v| v == 42.0));

    let fl = np::full_like(&f, 7.0);
    assert_eq!(fl.shape(), &[2, 3]);
    assert!(fl.iter().all(|&v| v == 7.0));

    let eye = np::eye::<f64>(3);
    assert_eq!(eye[[0, 0]], 1.0);
    assert_eq!(eye[[0, 1]], 0.0);
    assert_eq!(eye[[1, 1]], 1.0);

    // arange, linspace, logspace, geomspace
    let r = np::arange(0.0, 5.0, 1.0);
    assert_eq!(r, array![0.0, 1.0, 2.0, 3.0, 4.0]);

    let lin = np::linspace(0.0, 1.0, 5);
    assert_eq!(lin, array![0.0, 0.25, 0.5, 0.75, 1.0]);

    let log = np::logspace(0.0, 3.0, 4, 10.0);
    assert_eq!(log, array![1.0, 10.0, 100.0, 1000.0]);

    let geom = np::geomspace(1.0, 1000.0, 4);
    assert!(np::allclose(&geom, &array![1.0, 10.0, 100.0, 1000.0], 1e-5, 1e-5));

    // meshgrid
    let x = array![1.0, 2.0, 3.0];
    let y = array![4.0, 5.0];
    let (xx, yy) = np::meshgrid(&x, &y);
    assert_eq!(xx.shape(), &[2, 3]);
    assert_eq!(yy.shape(), &[2, 3]);
    assert_eq!(xx[[0, 0]], 1.0);
    assert_eq!(xx[[0, 2]], 3.0);
    assert_eq!(yy[[0, 0]], 4.0);
    assert_eq!(yy[[1, 0]], 5.0);

    // reshape, transpose, squeeze, ravel
    let mat = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let reshaped = np::reshape(&mat, (3, 2));
    assert_eq!(reshaped.shape(), &[3, 2]);

    let tr = np::transpose(&mat);
    assert_eq!(tr.shape(), &[3, 2]);
    assert_eq!(tr[[0, 1]], 4.0);

    let rav = np::ravel(&mat);
    assert_eq!(rav, array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let single_col = array![[1.0], [2.0], [3.0]];
    let sq = np::squeeze(&single_col);
    assert_eq!(sq, array![1.0, 2.0, 3.0]);

    // tile, repeat, roll, flip, pad
    let t = np::tile(&array![1.0, 2.0], 2);
    assert_eq!(t, array![1.0, 2.0, 1.0, 2.0]);

    let rep = np::repeat(&array![1.0, 2.0], 2);
    assert_eq!(rep, array![1.0, 1.0, 2.0, 2.0]);

    let rolled = np::roll(&array![1.0, 2.0, 3.0, 4.0, 5.0], 2);
    assert_eq!(rolled, array![4.0, 5.0, 1.0, 2.0, 3.0]);

    let flipped = np::flip(&array![1.0, 2.0, 3.0]);
    assert_eq!(flipped, array![3.0, 2.0, 1.0]);

    let padded = np::pad(&array![1.0, 2.0, 3.0], (2, 1), 0.0);
    assert_eq!(padded, array![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_np_math_and_stats() {
    let a = array![1.0, 4.0, 9.0, 16.0];
    let sqrt_a = np::sqrt(&a);
    assert_eq!(sqrt_a, array![1.0, 2.0, 3.0, 4.0]);

    let b = array![-1.5, 0.2, 2.7];
    assert_eq!(np::floor(&b), array![-2.0, 0.0, 2.0]);
    assert_eq!(np::ceil(&b), array![-1.0, 1.0, 3.0]);
    assert_eq!(np::round(&b), array![-2.0, 0.0, 3.0]);

    let x = array![1.0, 5.0, 3.0];
    let y = array![4.0, 2.0, 3.0];
    assert_eq!(np::maximum(&x, &y), array![4.0, 5.0, 3.0]);
    assert_eq!(np::minimum(&x, &y), array![1.0, 2.0, 3.0]);

    let data = array![10.0, 20.0, 30.0, 40.0, 50.0];
    assert_eq!(np::sum(&data), 150.0);
    assert_eq!(np::mean(&data), 30.0);
    assert_eq!(np::var(&data, 0.0), 200.0);
    assert!((np::std(&data, 0.0) - (200.0_f64).sqrt()).abs() < 1e-9);
    assert_eq!(np::min(&data), 10.0);
    assert_eq!(np::max(&data), 50.0);
    assert_eq!(np::argmin(&data), 0);
    assert_eq!(np::argmax(&data), 4);
    assert_eq!(np::median(&data), 30.0);
    assert_eq!(np::percentile(&data, 75.0), 40.0);
    assert_eq!(np::cumsum(&data), array![10.0, 30.0, 60.0, 100.0, 150.0]);
    assert_eq!(np::diff(&data, 1), array![10.0, 10.0, 10.0, 10.0]);

    // Covariance and correlation
    let xy = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]];
    let r = np::corrcoef(&xy).unwrap();
    assert!((r[[0, 1]] - 1.0).abs() < 1e-6);
}

#[test]
fn test_np_logic_and_sorting() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.000001, 1.999999, 3.000002];
    assert!(np::allclose(&a, &b, 1e-4, 1e-4));

    let mask = np::isclose(&a, &b, 1e-5, 1e-5);
    assert!(np::all(&mask));

    let bools = array![true, true, false];
    assert!(!np::all(&bools));
    assert!(np::any(&bools));

    let unsorted = array![3.0, 1.0, 4.0, 1.5, 2.0];
    let sorted = np::sort(&unsorted);
    assert_eq!(sorted, array![1.0, 1.5, 2.0, 3.0, 4.0]);

    let idxs = np::argsort(&unsorted);
    assert_eq!(idxs, array![1, 3, 4, 0, 2]);

    let u = np::unique(&array![2.0, 1.0, 2.0, 3.0, 1.0]);
    assert_eq!(u, array![1.0, 2.0, 3.0]);

    let cond = array![true, false, true];
    let selected = np::where_arr(&cond, &array![10.0, 20.0, 30.0], &array![1.0, 2.0, 3.0]);
    assert_eq!(selected, array![10.0, 2.0, 30.0]);
}

#[test]
fn test_np_linear_algebra() {
    let v1 = array![1.0, 2.0, 3.0];
    let v2 = array![4.0, 5.0, 6.0];
    assert_eq!(np::dot(&v1, &v2), 32.0);
    assert_eq!(np::inner(&v1, &v2), 32.0);

    let out = np::outer(&v1, &v2);
    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(out[[0, 0]], 4.0);
    assert_eq!(out[[2, 2]], 18.0);

    let m1 = array![[1.0, 2.0], [3.0, 4.0]];
    let m2 = array![[2.0, 0.0], [1.0, 2.0]];
    let prod = np::matmul(&m1, &m2);
    assert_eq!(prod, array![[4.0, 4.0], [10.0, 8.0]]);

    let c = np::cross(&array![1.0, 0.0, 0.0], &array![0.0, 1.0, 0.0]);
    assert_eq!(c, array![0.0, 0.0, 1.0]);

    let d = np::diag(&m1, 0);
    assert_eq!(d, array![1.0, 4.0]);
    assert_eq!(np::trace(&m1), 5.0);

    let df = np::diagflat(&array![1.0, 2.0], 0);
    assert_eq!(df, array![[1.0, 0.0], [0.0, 2.0]]);

    // linalg routines
    let det = np::linalg::det(&m1);
    assert!((det - (-2.0)).abs() < 1e-9);

    let inv = np::linalg::inv(&m1).unwrap();
    let id = np::matmul(&m1, &inv);
    assert!(np::allclose(&id, &np::eye(2), 1e-9, 1e-9));

    let norm_f = np::linalg::rsnorm(&v1, Some(2));
    assert!((norm_f - (14.0_f64).sqrt()).abs() < 1e-9);

    let rank = np::linalg::matrix_rank(&m1, None);
    assert_eq!(rank, 2);

    // solve Ax = b
    let b = array![5.0, 11.0];
    let x = np::linalg::solve_vec(&m1, &b).unwrap();
    assert!(np::allclose(&x, &array![1.0, 2.0], 1e-9, 1e-9));
}

#[test]
fn test_np_random() {
    let u = np::random::uniform(0.0, 10.0, &[100]);
    assert_eq!(u.shape(), &[100]);
    assert!(u.iter().all(|&v| (0.0..=10.0).contains(&v)));

    let n = np::random::normal(5.0, 1.0, &[500]);
    let mean = n.iter().sum::<f64>() / 500.0;
    assert!((mean - 5.0).abs() < 0.3);

    let r_int = np::random::randint(10, 20, 50);
    assert!(r_int.iter().all(|&v| (10..20).contains(&v)));

    let sampled = np::random::choice(&[10, 20, 30, 40], 2, false, None);
    assert_eq!(sampled.len(), 2);
    assert_ne!(sampled[0], sampled[1]);

    let perm = np::random::permutation(5);
    assert_eq!(perm.len(), 5);
}
