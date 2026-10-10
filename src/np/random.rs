use ndarray::{Array, Array2, ArrayD, IxDyn};
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal, Uniform};

/// Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
pub fn rand<Sh>(shape: Sh) -> Array<f64, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    Array::from_shape_simple_fn(shape, || dist.sample(&mut rng))
}

/// Create an array of the given shape and populate it with random samples from a standard normal distribution.
pub fn randn<Sh>(shape: Sh) -> Array<f64, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    let mut rng = rand::thread_rng();
    Array::from_shape_simple_fn(shape, || rng.sample(StandardNormal))
}

/// Create an array of the given shape and populate it with random integers from low (inclusive) to high (exclusive).
pub fn randint<Sh>(low: i32, high: i32, shape: Sh) -> Array<i32, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    Array::from_shape_simple_fn(shape, || dist.sample(&mut rng))
}

/// Helper to build an `ArrayD<f64>` from a shape slice and a generator closure.
fn array_from_shape_fn<F: FnMut() -> f64>(shape: &[usize], mut f: F) -> ArrayD<f64> {
    let dim = IxDyn(shape);
    let total: usize = shape.iter().product();
    let data: Vec<f64> = (0..total).map(|_| f()).collect();
    ArrayD::from_shape_vec(dim, data).unwrap()
}

/// Draw samples from a uniform distribution over `[low, high)`.
pub fn uniform(low: f64, high: f64, shape: &[usize]) -> ArrayD<f64> {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    array_from_shape_fn(shape, || dist.sample(&mut rng))
}

/// Draw samples from a normal distribution `N(mean, std^2)`.
pub fn normal(mean: f64, std: f64, shape: &[usize]) -> ArrayD<f64> {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::new(mean, std).unwrap();
    array_from_shape_fn(shape, || dist.sample(&mut rng))
}

/// Draw samples from an exponential distribution with the given `scale` (1/rate).
pub fn exponential(scale: f64, shape: &[usize]) -> ArrayD<f64> {
    assert!(scale > 0.0, "scale must be positive");
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Exp::new(1.0 / scale).unwrap();
    array_from_shape_fn(shape, || dist.sample(&mut rng))
}

/// Draw samples from a Poisson distribution.
///
/// Uses Knuth's algorithm for `lam < 30` and `rand_distr`'s rejection method for larger values.
pub fn poisson(lam: f64, shape: &[usize]) -> ArrayD<f64> {
    assert!(lam >= 0.0, "lambda must be non-negative");
    let mut rng = rand::thread_rng();
    if lam < 30.0 {
        array_from_shape_fn(shape, || poisson_knuth(lam, &mut rng) as f64)
    } else {
        let dist = rand_distr::Poisson::new(lam).unwrap();
        array_from_shape_fn(shape, || dist.sample(&mut rng) as f64)
    }
}

fn poisson_knuth<R: Rng>(lam: f64, rng: &mut R) -> u64 {
    if lam == 0.0 {
        return 0;
    }
    let limit = (-lam).exp();
    let mut k: u64 = 0;
    let mut p = 1.0_f64;
    loop {
        k += 1;
        p *= rng.r#gen::<f64>();
        if p <= limit {
            return k - 1;
        }
    }
}

/// Draw samples from a binomial distribution using the BTPE algorithm.
pub fn binomial(n: u64, p: f64, shape: &[usize]) -> ArrayD<f64> {
    assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
    let mut rng = rand::thread_rng();
    array_from_shape_fn(shape, || binomial_sample(n, p, &mut rng) as f64)
}

fn binomial_sample<R: Rng>(n: u64, p: f64, rng: &mut R) -> u64 {
    if n == 0 || p == 0.0 {
        return 0;
    }
    if p == 1.0 {
        return n;
    }

    let flipped = p > 0.5;
    let pp = if flipped { 1.0 - p } else { p };

    let nf = n as f64;
    let q = 1.0 - pp;
    let r = pp / q;
    let g = r * (nf + 1.0);
    let mut f_prev: f64;
    let mut p_prev: f64;

    if nf * pp < 30.0 {
        // Inverse transform for small n*p
        let mut f = q.powi(n as i32);
        let u: f64 = rng.r#gen();
        let mut x: u64 = 0;
        p_prev = f;
        while u > p_prev {
            x += 1;
            f *= g / x as f64 - r;
            f_prev = p_prev;
            p_prev = f_prev + f;
        }
        return if flipped { n - x } else { x };
    }

    // BTPE
    let np = nf * pp;
    let variance = np * q;
    let sigma = variance.sqrt();
    let c = 0.134 + 20.5 / (15.3 + nf * pp * q);
    let c_sigma = c * sigma;
    let a = (np + 0.5 - c_sigma).floor();
    let b = (np + 0.5 + c_sigma).ceil();

    let lambda_l = a.ln() - (nf - a + 1.0).ln() + (g / (nf - a + 1.0)).ln() - pp.ln();
    let lambda_r = (nf - a).ln() - (b + 1.0).ln() + ((b + 1.0) / g).ln() - q.ln();

    let p1 = 2.0 * c_sigma + 1.0;
    let p2 = a - lambda_l * a;
    let _p3 = b + 1.0 + lambda_r * (nf - b);
    let p4 = (1.0 + lambda_l) * a;
    let p5 = (nf - b) * (1.0 + lambda_r);

    loop {
        let u: f64 = rng.r#gen();
        let v: f64 = rng.r#gen::<f64>() * p1;
        let u2 = u - 0.5;
        let k_float = ((2.0 * a / p1 + u2) * c_sigma + np + 0.5).floor();
        if k_float < a || k_float > b {
            continue;
        }
        let k = k_float as u64;

        let accept = if v < p4 {
            v <= p2 + lambda_l * k_float
        } else if v <= p5 {
            v <= (nf - k_float) * lambda_r - (nf - k_float + 1.0).ln() + (nf - b).ln()
        } else {
            let f_k = lambda_l * k_float.ln() - (nf - k_float + 1.0).ln();
            let f_m = lambda_l * a.ln() - (nf - a + 1.0).ln();
            v <= f_k - f_m + p2
        };

        if !accept {
            let lhs = (k_float + 0.5) * (k_float + 0.5).ln() - k_float - 0.9189385332;
            let rhs = v - p2;
            if rhs > lhs {
                let x1 = k_float;
                let x2 = (nf - k_float + 1.0).ln();
                let x3 = (nf - a + 1.0).ln();
                let x4 = (a).ln();
                let log_lhs = x1 * lambda_l - x2 - (x4 * lambda_l - x3);
                if rhs > log_lhs {
                    continue;
                }
            }
        }
        return if flipped { n - k } else { k };
    }
}

/// Draw samples from a Beta distribution using the gamma method.
pub fn beta(a: f64, b: f64, shape: &[usize]) -> ArrayD<f64> {
    assert!(a > 0.0, "a must be positive");
    assert!(b > 0.0, "b must be positive");
    let mut rng = rand::thread_rng();
    array_from_shape_fn(shape, || {
        let x = gamma_sample(a, 1.0, &mut rng);
        let y = gamma_sample(b, 1.0, &mut rng);
        x / (x + y)
    })
}

/// Draw samples from a Gamma distribution using Marsaglia and Tsang's method.
pub fn gamma(shape_param: f64, scale: f64, size: &[usize]) -> ArrayD<f64> {
    assert!(shape_param > 0.0, "shape must be positive");
    assert!(scale > 0.0, "scale must be positive");
    let mut rng = rand::thread_rng();
    array_from_shape_fn(size, || gamma_sample(shape_param, scale, &mut rng))
}

fn gamma_sample<R: Rng>(shape_param: f64, scale: f64, rng: &mut R) -> f64 {
    if shape_param < 1.0 {
        let u: f64 = rng.r#gen();
        gamma_sample(shape_param + 1.0, scale, rng) * u.powf(1.0 / shape_param)
    } else {
        let d = shape_param - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let mut x: f64;
            let mut v;
            loop {
                x = rng.sample(StandardNormal);
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            v = v * v * v;
            let u: f64 = rng.r#gen();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v * scale;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }
}

/// Draw samples from a multivariate normal distribution.
pub fn multivariate_normal(mean: &[f64], cov: &Array2<f64>, size: usize) -> Array2<f64> {
    let n = mean.len();
    assert_eq!(cov.dim(), (n, n), "cov must be (n, n) where n = mean.len()");

    let l = cholesky(cov);
    let mut rng = rand::thread_rng();
    let mut result = Array2::zeros((size, n));

    for i in 0..size {
        let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
        for j in 0..n {
            let mut val = mean[j];
            for k in 0..=j {
                val += l[[j, k]] * z[k];
            }
            result[[i, j]] = val;
        }
    }
    result
}

/// Cholesky decomposition (lower triangular).
fn cholesky(m: &Array2<f64>) -> Array2<f64> {
    let n = m.dim().0;
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = m[[i, i]] - sum;
                assert!(diag > 0.0, "cov matrix is not positive definite");
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (m[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    l
}

/// Random sampling from a slice of indices.
///
/// If `replace` is false, samples without replacement (all returned values are unique).
/// If `p` is `Some`, uses weighted sampling.
pub fn choice(a: &[usize], size: usize, replace: bool, p: Option<&[f64]>) -> Vec<usize> {
    let mut rng = rand::thread_rng();

    if !replace {
        assert!(size <= a.len(), "cannot sample more than {} without replacement", a.len());
    }

    if let Some(weights) = p {
        assert_eq!(weights.len(), a.len(), "weights length must match a");
        let total: f64 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "weights must sum to 1");

        if replace {
            let cdf = build_cdf(weights);
            (0..size)
                .map(|_| {
                    let u: f64 = rng.r#gen();
                    let idx = cdf.partition_point(|&v| v < u);
                    a[idx.min(a.len() - 1)]
                })
                .collect()
        } else {
            let mut remaining: Vec<usize> = (0..a.len()).collect();
            let mut w: Vec<f64> = weights.to_vec();
            let mut result = Vec::with_capacity(size);
            for _ in 0..size {
                let wt: f64 = w.iter().sum();
                let normalized: Vec<f64> = w.iter().map(|v| v / wt).collect();
                let cdf = build_cdf(&normalized);
                let u: f64 = rng.r#gen();
                let idx = cdf.partition_point(|&v| v < u).min(remaining.len() - 1);
                result.push(a[remaining[idx]]);
                remaining.swap_remove(idx);
                w.swap_remove(idx);
            }
            result
        }
    } else {
        if replace {
            let dist = Uniform::new(0, a.len());
            (0..size).map(|_| a[dist.sample(&mut rng)]).collect()
        } else {
            let mut indices: Vec<usize> = (0..a.len()).collect();
            indices.shuffle(&mut rng);
            indices[..size].iter().map(|&i| a[i]).collect()
        }
    }
}

fn build_cdf(weights: &[f64]) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(weights.len());
    let mut cumulative = 0.0;
    for &w in weights {
        cumulative += w;
        cdf.push(cumulative);
    }
    cdf
}

/// In-place Fisher-Yates shuffle.
pub fn shuffle(x: &mut [usize]) {
    let mut rng = rand::thread_rng();
    for i in (1..x.len()).rev() {
        let j = rng.gen_range(0..=i);
        x.swap(i, j);
    }
}

/// Return a random permutation of `0..n`.
pub fn permutation(n: usize) -> Vec<usize> {
    let mut x: Vec<usize> = (0..n).collect();
    shuffle(&mut x);
    x
}

/// Create a seeded RNG for reproducible sequences.
///
/// Note: `rand 0.8`'s `thread_rng()` cannot be seeded globally.
/// This function returns a `StdRng` that can be used explicitly.
pub fn seed(s: u64) -> StdRng {
    StdRng::seed_from_u64(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 100_000;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_uniform_range() {
        let arr = uniform(0.0, 1.0, &[N]);
        for &v in arr.iter() {
            assert!((0.0..1.0).contains(&v), "value {v} out of [0,1)");
        }
        let mean = arr.mean().unwrap();
        assert!(approx_eq(mean, 0.5, 0.02), "mean {mean} not near 0.5");
    }

    #[test]
    fn test_uniform_custom_range() {
        let arr = uniform(2.0, 5.0, &[N]);
        for &v in arr.iter() {
            assert!((2.0..5.0).contains(&v), "value {v} out of [2,5)");
        }
        let mean = arr.mean().unwrap();
        assert!(approx_eq(mean, 3.5, 0.05), "mean {mean} not near 3.5");
    }

    #[test]
    fn test_normal_shape() {
        let arr = normal(0.0, 1.0, &[N]);
        let mean = arr.mean().unwrap();
        let std = arr.std(0.0);
        assert!(approx_eq(mean, 0.0, 0.05), "mean {mean} not near 0.0");
        assert!(approx_eq(std, 1.0, 0.05), "std {std} not near 1.0");
    }

    #[test]
    fn test_exponential() {
        let arr = exponential(2.0, &[N]);
        for &v in arr.iter() {
            assert!(v >= 0.0, "value {v} negative");
        }
        let mean = arr.mean().unwrap();
        assert!(approx_eq(mean, 2.0, 0.1), "mean {mean} not near 2.0");
    }

    #[test]
    fn test_poisson_small_lambda() {
        let lam = 5.0;
        let arr = poisson(lam, &[N]);
        let mean = arr.mean().unwrap();
        assert!(approx_eq(mean, lam, 0.1), "mean {mean} not near {lam}");
        for &v in arr.iter() {
            assert!(v >= 0.0, "value {v} negative");
            assert!(v == v.floor(), "value {v} not integer-valued");
        }
    }

    #[test]
    fn test_poisson_large_lambda() {
        let lam = 100.0;
        let arr = poisson(lam, &[N]);
        let mean = arr.mean().unwrap();
        assert!(approx_eq(mean, lam, 1.0), "mean {mean} not near {lam}");
    }

    #[test]
    fn test_poisson_zero() {
        let arr = poisson(0.0, &[10]);
        for &v in arr.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_binomial() {
        let n = 20;
        let p = 0.3;
        let arr = binomial(n, p, &[N]);
        let mean = arr.mean().unwrap();
        let expected = n as f64 * p;
        assert!(approx_eq(mean, expected, 0.2), "mean {mean} not near {expected}");
        for &v in arr.iter() {
            assert!(v >= 0.0 && v <= n as f64, "value {v} out of [0, {n}]");
            assert!(v == v.floor(), "value {v} not integer-valued");
        }
    }

    #[test]
    fn test_binomial_extreme_p() {
        let arr0 = binomial(10, 0.0, &[100]);
        for &v in arr0.iter() {
            assert_eq!(v, 0.0);
        }
        let arr1 = binomial(10, 1.0, &[100]);
        for &v in arr1.iter() {
            assert_eq!(v, 10.0);
        }
    }

    #[test]
    fn test_beta() {
        let arr = beta(2.0, 5.0, &[N]);
        for &v in arr.iter() {
            assert!((0.0..1.0).contains(&v), "value {v} out of [0,1)");
        }
        let mean = arr.mean().unwrap();
        let expected = 2.0 / (2.0 + 5.0);
        assert!(approx_eq(mean, expected, 0.02), "mean {mean} not near {expected}");
    }

    #[test]
    fn test_gamma() {
        let arr = gamma(2.0, 3.0, &[N]);
        for &v in arr.iter() {
            assert!(v >= 0.0, "value {v} negative");
        }
        let mean = arr.mean().unwrap();
        let expected = 2.0 * 3.0;
        assert!(approx_eq(mean, expected, 0.2), "mean {mean} not near {expected}");
    }

    #[test]
    fn test_multivariate_normal() {
        let mean = vec![1.0, 2.0];
        let cov = ndarray::arr2(&[[1.0, 0.5], [0.5, 2.0]]);
        let arr = multivariate_normal(&mean, &cov, N);
        assert_eq!(arr.dim(), (N, 2));

        let m0 = arr.column(0).mean().unwrap();
        let m1 = arr.column(1).mean().unwrap();
        assert!(approx_eq(m0, 1.0, 0.05), "mean0 {m0} not near 1.0");
        assert!(approx_eq(m1, 2.0, 0.05), "mean1 {m1} not near 2.0");

        let v0 = arr.column(0).var(0.0);
        let v1 = arr.column(1).var(0.0);
        assert!(approx_eq(v0, 1.0, 0.1), "var0 {v0} not near 1.0");
        assert!(approx_eq(v1, 2.0, 0.2), "var1 {v1} not near 2.0");
    }

    #[test]
    fn test_choice_with_replacement() {
        let a: Vec<usize> = (0..10).collect();
        let result = choice(&a, 1000, true, None);
        assert_eq!(result.len(), 1000);
        for &v in &result {
            assert!(v < 10, "value {v} out of range");
        }
    }

    #[test]
    fn test_choice_without_replacement() {
        let a: Vec<usize> = (0..20).collect();
        let result = choice(&a, 10, false, None);
        assert_eq!(result.len(), 10);
        let mut sorted = result.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10, "duplicates found in without-replacement sample");
    }

    #[test]
    fn test_choice_weighted() {
        let a: Vec<usize> = vec![0, 1];
        let p = vec![1.0, 0.0];
        let result = choice(&a, 100, true, Some(&p));
        for &v in &result {
            assert_eq!(v, 0, "expected all zeros with p=[1,0]");
        }
    }

    #[test]
    fn test_choice_weighted_without_replacement() {
        let a: Vec<usize> = (0..5).collect();
        let p = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        let result = choice(&a, 3, false, Some(&p));
        assert_eq!(result.len(), 3);
        let mut sorted = result.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "duplicates in weighted without-replacement");
    }

    #[test]
    fn test_shuffle_is_permutation() {
        let n = 100;
        let mut x: Vec<usize> = (0..n).collect();
        shuffle(&mut x);
        let mut sorted = x.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(sorted, expected, "shuffle did not produce a valid permutation");
    }

    #[test]
    fn test_permutation() {
        let n = 100;
        let p = permutation(n);
        assert_eq!(p.len(), n);
        let mut sorted = p.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(sorted, expected, "permutation not valid");
    }

    #[test]
    fn test_seed_reproducibility() {
        let mut rng1 = seed(42);
        let mut rng2 = seed(42);
        for _ in 0..100 {
            assert_eq!(rng1.r#gen::<f64>(), rng2.r#gen::<f64>());
        }
    }

    #[test]
    fn test_array_shapes() {
        let arr = uniform(0.0, 1.0, &[2, 3, 4]);
        assert_eq!(arr.shape(), &[2, 3, 4]);

        let arr = normal(0.0, 1.0, &[5, 5]);
        assert_eq!(arr.shape(), &[5, 5]);

        let arr = gamma(1.0, 1.0, &[10]);
        assert_eq!(arr.shape(), &[10]);
    }

    #[test]
    fn test_cholesky_identity() {
        let eye = ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let l = cholesky(&eye);
        assert!(approx_eq(l[[0, 0]], 1.0, 1e-10));
        assert!(approx_eq(l[[1, 0]], 0.0, 1e-10));
        assert!(approx_eq(l[[1, 1]], 1.0, 1e-10));
    }
}
