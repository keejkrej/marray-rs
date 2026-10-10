use ndarray::{Array1, Array2};

/// Standard error function approximation (Abramowitz and Stegun).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp());
    sign * y
}

pub struct Norm;

impl Norm {
    /// Probability density function of normal distribution.
    pub fn pdf(x: f64, loc: f64, scale: f64) -> f64 {
        let variance = scale * scale;
        let exponent = -((x - loc).powi(2)) / (2.0 * variance);
        (1.0 / (scale * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
    }

    /// Cumulative distribution function of normal distribution.
    pub fn cdf(x: f64, loc: f64, scale: f64) -> f64 {
        0.5 * (1.0 + erf((x - loc) / (scale * 2.0_f64.sqrt())))
    }
}

/// Compute Pearson correlation coefficient and p-value.
pub fn pearsonr(x: &Array1<f64>, y: &Array1<f64>) -> Result<(f64, f64), String> {
    let n = x.len();
    if n != y.len() {
        return Err("Arrays must have the same length".to_string());
    }
    if n == 0 {
        return Err("Arrays must not be empty".to_string());
    }
    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let den = (den_x * den_y).sqrt();
    if den < 1e-14 {
        return Ok((0.0, 1.0));
    }
    let r = num / den;

    let p_value = if n <= 2 {
        1.0
    } else if (1.0 - r * r) <= 1e-14 {
        0.0
    } else {
        let t_stat = r * ((n - 2) as f64 / (1.0 - r * r)).sqrt();
        let norm_cdf = Norm::cdf(t_stat.abs(), 0.0, 1.0);
        (2.0 * (1.0 - norm_cdf)).min(1.0).max(0.0)
    };

    Ok((r, p_value))
}

/// Compute ranks for data, handling ties by averaging ranks.
fn rankdata(arr: &Array1<f64>) -> Array1<f64> {
    let n = arr.len();
    let mut indexed: Vec<(usize, f64)> = arr.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        let rank_sum: f64 = (i + 1..=j).map(|r| r as f64).sum();
        let avg_rank = rank_sum / ((j - i) as f64);
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    Array1::from_vec(ranks)
}

/// Compute Spearman rank correlation coefficient and p-value.
pub fn spearmanr(x: &Array1<f64>, y: &Array1<f64>) -> Result<(f64, f64), String> {
    let rx = rankdata(x);
    let ry = rankdata(y);
    pearsonr(&rx, &ry)
}

/// Compute standard score (z-score) of each value in the sample.
pub fn zscore(arr: &Array1<f64>, ddof: f64) -> Array1<f64> {
    let n = arr.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let mean = arr.mean().unwrap_or(0.0);

    let mut sum_sq = 0.0;
    for &val in arr.iter() {
        sum_sq += (val - mean).powi(2);
    }
    let div = (n as f64) - ddof;
    let variance = if div > 0.0 { sum_sq / div } else { 0.0 };
    let std = if variance > 1e-14 { variance.sqrt() } else { 1.0 };

    arr.mapv(|val| (val - mean) / std)
}

// ──────────────────────────────────────────────────────────────────────────────
// Special functions
// ──────────────────────────────────────────────────────────────────────────────

/// Log of the gamma function (Lanczos approximation).
pub fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln();
        return s - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let mut sum = c[0];
    for i in 1..c.len() {
        sum += c[i] / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Gamma function.
pub fn gamma(x: f64) -> f64 {
    lgamma(x).exp()
}

/// Log of the beta function.
pub fn lbeta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Beta function.
pub fn beta(a: f64, b: f64) -> f64 {
    lbeta(a, b).exp()
}

/// Regularized lower incomplete gamma function P(a, x) = γ(a,x) / Γ(a).
pub fn gammainc(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series expansion
        gammainc_series(a, x)
    } else {
        // Continued fraction for Q(a,x) = 1 - P(a,x)
        1.0 - gammainc_cf(a, x)
    }
}

/// Series expansion for regularized lower incomplete gamma.
fn gammainc_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..300 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-14 {
            break;
        }
    }
    sum * (-x + a * x.ln() - lgamma(a)).exp()
}

/// Continued fraction for regularized upper incomplete gamma Q(a,x) = Γ(a,x)/Γ(a).
/// Uses Lentz's method (Numerical Recipes).
fn gammainc_cf(a: f64, x: f64) -> f64 {
    let eps = 1e-14;
    let it_max = 300;

    let b = x + 1.0 - a;
    let mut c = 1.0e30;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=it_max {
        let an = -(i as f64) * (a - i as f64);
        let bi = b + 2.0 * i as f64;
        d = an * d + bi;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = bi + an / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let del = c * d;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    let prefactor = (-x + a * x.ln() - lgamma(a)).exp();
    prefactor * h
}

/// Regularized incomplete beta function I_x(a, b).
/// Uses continued fraction (Lentz's method).
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betainc_cf(a, b, x) / a + 1e-15 // small epsilon for numerical stability
    } else {
        1.0 - bt * betainc_cf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for regularized incomplete beta.
fn betainc_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m2 = 2 * m;
        // even step
        let aa = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        h *= d * c;

        // odd step
        let aa2 = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa2 * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + aa2 / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// Bisection search for quantile function (ppf).
pub fn bisect_ppf(
    cdf_fn: impl Fn(f64) -> f64,
    p: f64,
    lo: f64,
    hi: f64,
    tol: f64,
    max_iter: usize,
) -> f64 {
    let mut lo = lo;
    let mut hi = hi;
    for _ in 0..max_iter {
        let mid = (lo + hi) / 2.0;
        let val = cdf_fn(mid);
        if (hi - lo).abs() < tol {
            return mid;
        }
        if val < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Inverse of the standard normal CDF (rational approximation, Abramowitz & Stegun 26.2.23).
pub fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let t: f64;
    if p < 0.5 {
        t = (-2.0 * p.ln()).sqrt();
    } else {
        t = (-2.0 * (1.0 - p).ln()).sqrt();
    }
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 {
        -z
    } else {
        z
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Distribution traits
// ──────────────────────────────────────────────────────────────────────────────

pub trait ContinuousDistribution {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn ppf(&self, p: f64) -> f64;
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    fn mean(&self) -> f64;
    fn var(&self) -> f64;
    fn std(&self) -> f64;
    fn rvs(&self, n: usize) -> Vec<f64>;
}

pub trait DiscreteDistribution {
    fn pmf(&self, k: u64) -> f64;
    fn cdf(&self, k: u64) -> f64;
    fn mean(&self) -> f64;
    fn var(&self) -> f64;
}

// ──────────────────────────────────────────────────────────────────────────────
// Normal distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct NormDist {
    pub mean: f64,
    pub std: f64,
}

impl NormDist {
    pub fn new(mean: f64, std: f64) -> Self {
        NormDist { mean, std }
    }
    pub fn standard() -> Self {
        NormDist { mean: 0.0, std: 1.0 }
    }
}

impl ContinuousDistribution for NormDist {
    fn pdf(&self, x: f64) -> f64 {
        Norm::pdf(x, self.mean, self.std)
    }

    fn cdf(&self, x: f64) -> f64 {
        Norm::cdf(x, self.mean, self.std)
    }

    fn ppf(&self, p: f64) -> f64 {
        self.mean + self.std * norm_ppf(p)
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn var(&self) -> f64 {
        self.std * self.std
    }

    fn std(&self) -> f64 {
        self.std
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand_distr::Distribution as RandDist;
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(self.mean, self.std).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Student's t distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct TDist {
    pub df: f64,
}

impl TDist {
    pub fn new(df: f64) -> Self {
        TDist { df }
    }
}

impl ContinuousDistribution for TDist {
    fn pdf(&self, t: f64) -> f64 {
        let df = self.df;
        let num = lgamma((df + 1.0) / 2.0);
        let den = (df * std::f64::consts::PI).sqrt() * lgamma(df / 2.0).exp() ;
        let coeff = num.exp() / den;
        coeff * (1.0 + t * t / df).powf(-(df + 1.0) / 2.0)
    }

    fn cdf(&self, t: f64) -> f64 {
        let df = self.df;
        let x = df / (df + t * t);
        let ibeta = betainc(df / 2.0, 0.5, x);
        if t >= 0.0 {
            1.0 - 0.5 * ibeta
        } else {
            0.5 * ibeta
        }
    }

    fn ppf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        // Use normal approximation as initial guess
        let z = norm_ppf(p);
        // Refine with bisection
        let lo = if p < 0.5 { -1000.0 } else { z - 100.0 };
        let hi = if p < 0.5 { z + 100.0 } else { 1000.0 };
        bisect_ppf(|t| self.cdf(t), p, lo, hi, 1e-12, 200)
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn mean(&self) -> f64 {
        if self.df > 1.0 { 0.0 } else { f64::NAN }
    }

    fn var(&self) -> f64 {
        if self.df > 2.0 {
            self.df / (self.df - 2.0)
        } else if self.df > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }

    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand_distr::Distribution as RandDist;
        let mut rng = rand::thread_rng();
        let dist = rand_distr::StudentT::new(self.df).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Chi-squared distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct Chi2Dist {
    pub df: f64,
}

impl Chi2Dist {
    pub fn new(df: f64) -> Self {
        Chi2Dist { df }
    }
}

impl ContinuousDistribution for Chi2Dist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let k = self.df;
        let num = (k / 2.0 - 1.0) * x.ln() - x / 2.0;
        let den = (k / 2.0) * 2.0_f64.ln() + lgamma(k / 2.0);
        (num - den).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        gammainc(self.df / 2.0, x / 2.0)
    }

    fn ppf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        // Use normal approximation as initial guess
        let z = norm_ppf(p);
        // Wilson-Hilferty approximation for initial guess
        let k = self.df;
        let guess = k * (1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt()).powi(3);
        let lo = 0.0;
        let hi = guess.max(k * 10.0);
        bisect_ppf(|x| self.cdf(x), p, lo, hi, 1e-12, 200)
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn mean(&self) -> f64 {
        self.df
    }

    fn var(&self) -> f64 {
        2.0 * self.df
    }

    fn std(&self) -> f64 {
        (2.0 * self.df).sqrt()
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand_distr::Distribution as RandDist;
        let mut rng = rand::thread_rng();
        let dist = rand_distr::ChiSquared::new(self.df).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// F distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct FDist {
    pub dfn: f64,
    pub dfd: f64,
}

impl FDist {
    pub fn new(dfn: f64, dfd: f64) -> Self {
        FDist { dfn, dfd }
    }
}

impl ContinuousDistribution for FDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let d1 = self.dfn;
        let d2 = self.dfd;
        let num = lgamma((d1 + d2) / 2.0) + (d1 / 2.0) * d1.ln() + (d2 / 2.0) * d2.ln()
            - lgamma(d1 / 2.0)
            - lgamma(d2 / 2.0)
            + (d1 / 2.0 - 1.0) * x.ln()
            - ((d1 + d2) / 2.0) * (1.0 + d1 * x / d2).ln();
        num.exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let d1 = self.dfn;
        let d2 = self.dfd;
        let z = d1 * x / (d1 * x + d2);
        betainc(d1 / 2.0, d2 / 2.0, z)
    }

    fn ppf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        // Use bisection with a reasonable range
        let lo = 0.0;
        let hi = 1000.0;
        bisect_ppf(|x| self.cdf(x), p, lo, hi, 1e-12, 200)
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn mean(&self) -> f64 {
        if self.dfd > 2.0 {
            self.dfd / (self.dfd - 2.0)
        } else {
            f64::NAN
        }
    }

    fn var(&self) -> f64 {
        if self.dfd > 4.0 {
            2.0 * self.dfd * self.dfd * (self.dfn + self.dfd - 2.0)
                / (self.dfn * (self.dfd - 2.0).powi(2) * (self.dfd - 4.0))
        } else {
            f64::NAN
        }
    }

    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand_distr::Distribution as RandDist;
        let mut rng = rand::thread_rng();
        let dist = rand_distr::FisherF::new(self.dfn, self.dfd).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Beta distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct BetaDist {
    pub a: f64,
    pub b: f64,
}

impl BetaDist {
    pub fn new(a: f64, b: f64) -> Self {
        BetaDist { a, b }
    }
}

impl ContinuousDistribution for BetaDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        let num = (self.a - 1.0) * x.ln() + (self.b - 1.0) * (1.0 - x).ln();
        let den = lbeta(self.a, self.b);
        (num - den).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        betainc(self.a, self.b, x)
    }

    fn ppf(&self, p: f64) -> f64 {
        bisect_ppf(|x| self.cdf(x), p, 0.0, 1.0, 1e-12, 200)
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }

    fn var(&self) -> f64 {
        let ab = self.a + self.b;
        self.a * self.b / (ab * ab * (ab + 1.0))
    }

    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand_distr::Distribution as RandDist;
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Beta::new(self.a, self.b).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Gamma distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct GammaDist {
    pub shape: f64,
    pub scale: f64,
}

impl GammaDist {
    pub fn new(shape: f64, scale: f64) -> Self {
        GammaDist { shape, scale }
    }
}

impl ContinuousDistribution for GammaDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let k = self.shape;
        let theta = self.scale;
        let num = (k - 1.0) * x.ln() - x / theta;
        let den = k * theta.ln() + lgamma(k);
        (num - den).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        gammainc(self.shape, x / self.scale)
    }

    fn ppf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        // Use mean as initial guess center
        let lo = 0.0;
        let hi = self.shape * self.scale * 10.0;
        bisect_ppf(|x| self.cdf(x), p, lo, hi, 1e-12, 200)
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn mean(&self) -> f64 {
        self.shape * self.scale
    }

    fn var(&self) -> f64 {
        self.shape * self.scale * self.scale
    }

    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand_distr::Distribution as RandDist;
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Gamma::new(self.shape, self.scale).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Exponential distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct ExponentialDist {
    pub scale: f64,
}

impl ExponentialDist {
    pub fn new(scale: f64) -> Self {
        ExponentialDist { scale }
    }
}

impl ContinuousDistribution for ExponentialDist {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        (-x / self.scale).exp() / self.scale
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        1.0 - (-x / self.scale).exp()
    }

    fn ppf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        -self.scale * (1.0 - p).ln()
    }

    fn sf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 1.0;
        }
        (-x / self.scale).exp()
    }

    fn mean(&self) -> f64 {
        self.scale
    }

    fn var(&self) -> f64 {
        self.scale * self.scale
    }

    fn std(&self) -> f64 {
        self.scale
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand_distr::Distribution as RandDist;
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Exp::new(1.0 / self.scale).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Uniform distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct UniformDist {
    pub loc: f64,
    pub scale: f64,
}

impl UniformDist {
    pub fn new(loc: f64, scale: f64) -> Self {
        UniformDist { loc, scale }
    }
    pub fn standard() -> Self {
        UniformDist { loc: 0.0, scale: 1.0 }
    }
}

impl ContinuousDistribution for UniformDist {
    fn pdf(&self, x: f64) -> f64 {
        if x >= self.loc && x <= self.loc + self.scale {
            1.0 / self.scale
        } else {
            0.0
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.loc {
            0.0
        } else if x > self.loc + self.scale {
            1.0
        } else {
            (x - self.loc) / self.scale
        }
    }

    fn ppf(&self, p: f64) -> f64 {
        self.loc + p * self.scale
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn mean(&self) -> f64 {
        self.loc + self.scale / 2.0
    }

    fn var(&self) -> f64 {
        self.scale * self.scale / 12.0
    }

    fn std(&self) -> f64 {
        self.scale / 12.0_f64.sqrt()
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen_range(self.loc..self.loc + self.scale)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Log-normal distribution
// ──────────────────────────────────────────────────────────────────────────────

pub struct LogNormalDist {
    pub mean: f64,
    pub sigma: f64,
}

impl LogNormalDist {
    pub fn new(mean: f64, sigma: f64) -> Self {
        LogNormalDist { mean, sigma }
    }
}

impl ContinuousDistribution for LogNormalDist {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let mu = self.mean;
        let s = self.sigma;
        let num = -(x.ln() - mu).powi(2);
        let den = 2.0 * s * s;
        let coeff = 1.0 / (x * s * (2.0 * std::f64::consts::PI).sqrt());
        coeff * (num / den).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        Norm::cdf(x.ln(), self.mean, self.sigma)
    }

    fn ppf(&self, p: f64) -> f64 {
        (self.mean + self.sigma * norm_ppf(p)).exp()
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn mean(&self) -> f64 {
        (self.mean + self.sigma * self.sigma / 2.0).exp()
    }

    fn var(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        ((s2).exp() - 1.0) * (2.0 * self.mean + s2).exp()
    }

    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    fn rvs(&self, n: usize) -> Vec<f64> {
        use rand_distr::Distribution as RandDist;
        let mut rng = rand::thread_rng();
        let dist = rand_distr::LogNormal::new(self.mean, self.sigma).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Bernoulli distribution (discrete)
// ──────────────────────────────────────────────────────────────────────────────

pub struct BernoulliDist {
    pub p: f64,
}

impl BernoulliDist {
    pub fn new(p: f64) -> Self {
        BernoulliDist { p }
    }
}

impl DiscreteDistribution for BernoulliDist {
    fn pmf(&self, k: u64) -> f64 {
        match k {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }

    fn cdf(&self, k: u64) -> f64 {
        if k < 1 {
            1.0 - self.p
        } else {
            1.0
        }
    }

    fn mean(&self) -> f64 {
        self.p
    }

    fn var(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Descriptive statistics
// ──────────────────────────────────────────────────────────────────────────────

/// Excess kurtosis. If `fisher` is true, subtracts 3 (Fisher's definition).
pub fn kurtosis(x: &[f64], fisher: bool) -> f64 {
    let n = x.len();
    if n < 4 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let m2: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
    let m4: f64 = x.iter().map(|v| (v - mean).powi(4)).sum();
    let s2 = m2 / (n - 1) as f64;
    let n_f = n as f64;
    let k = (n_f * (n_f + 1.0)) / ((n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0))
        * m4 / (s2 * s2)
        - 3.0 * (n_f - 1.0).powi(2) / ((n_f - 2.0) * (n_f - 3.0));
    if fisher { k } else { k + 3.0 }
}

/// Skewness (Fisher's unbiased version).
pub fn skew(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let m2: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
    let m3: f64 = x.iter().map(|v| (v - mean).powi(3)).sum();
    let s = (m2 / (n - 1) as f64).sqrt();
    // Fisher's unbiased skewness
    let n_f = n as f64;
    (n_f / ((n_f - 1.0) * (n_f - 2.0))) * m3 / (s * s * s)
}

/// Returns (nobs, min, max, mean, variance, skewness, kurtosis).
pub fn describe(x: &[f64]) -> (usize, f64, f64, f64, f64, f64, f64) {
    let n = x.len();
    if n == 0 {
        return (0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        f64::NAN
    };
    let sk = skew(x);
    let ku = kurtosis(x, true);
    (n, min, max, mean, var, sk, ku)
}

/// Returns (mode_value, count). Uses histogram binning for continuous data.
pub fn mode(x: &[f64]) -> (f64, usize) {
    use std::collections::HashMap;
    let mut counts: HashMap<String, (f64, usize)> = HashMap::new();
    // Round to 12 decimal places for grouping
    for &v in x {
        let key = format!("{:.12}", v);
        let entry = counts.entry(key).or_insert((v, 0));
        entry.1 += 1;
    }
    counts
        .values()
        .max_by_key(|&&(_, c)| c)
        .map(|&(v, c)| (v, c))
        .unwrap_or((f64::NAN, 0))
}

/// Interquartile range (75th - 25th percentile).
pub fn iqr(x: &[f64]) -> f64 {
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let q1_idx = 0.25 * (n - 1) as f64;
    let q3_idx = 0.75 * (n - 1) as f64;
    let q1 = interp_index(&sorted, q1_idx);
    let q3 = interp_index(&sorted, q3_idx);
    q3 - q1
}

/// Standard error of mean.
pub fn sem(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (var / n as f64).sqrt()
}

/// Helper: linear interpolation at fractional index.
fn interp_index(sorted: &[f64], idx: f64) -> f64 {
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi || hi >= sorted.len() {
        sorted[lo.min(sorted.len() - 1)]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Linear regression. Returns (slope, intercept, r_value, p_value, std_err).
pub fn linregress(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64, f64, f64), String> {
    let n = x.len();
    if n != y.len() {
        return Err("Arrays must have the same length".to_string());
    }
    if n < 3 {
        return Err("Need at least 3 data points".to_string());
    }
    let x_mean = x.iter().sum::<f64>() / n as f64;
    let y_mean = y.iter().sum::<f64>() / n as f64;

    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    let mut ss_yy = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        ss_xx += dx * dx;
        ss_xy += dx * dy;
        ss_yy += dy * dy;
    }

    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;
    let r_value = if ss_xx > 0.0 && ss_yy > 0.0 {
        ss_xy / (ss_xx * ss_yy).sqrt()
    } else {
        0.0
    };

    // t-statistic for slope significance
    let ss_res: f64 = (0..n).map(|i| (y[i] - (slope * x[i] + intercept)).powi(2)).sum();
    let se_slope = (ss_res / (n - 2) as f64 / ss_xx).sqrt();
    let t_stat = slope / se_slope;
    let t_dist = TDist::new((n - 2) as f64);
    let p_value = 2.0 * t_dist.sf(t_stat.abs());

    Ok((slope, intercept, r_value, p_value, se_slope))
}

/// Shannon entropy: H = -sum(pk * log(pk)).
pub fn entropy(pk: &[f64]) -> f64 {
    -pk.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f64>()
}

// ──────────────────────────────────────────────────────────────────────────────
// Hypothesis tests
// ──────────────────────────────────────────────────────────────────────────────

/// One-sample t-test. Returns (t-statistic, two-tailed p-value).
pub fn ttest_1samp(x: &[f64], popmean: f64) -> (f64, f64) {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    let t = (mean - popmean) / se;
    let t_dist = TDist::new((n - 1) as f64);
    let p = 2.0 * t_dist.sf(t.abs());
    (t, p)
}

/// Independent two-sample t-test. Returns (t-statistic, two-tailed p-value).
/// If `equal_var` is true, uses pooled variance; otherwise Welch's.
pub fn ttest_ind(x: &[f64], y: &[f64], equal_var: bool) -> (f64, f64) {
    let n1 = x.len();
    let n2 = y.len();
    let m1 = x.iter().sum::<f64>() / n1 as f64;
    let m2 = y.iter().sum::<f64>() / n2 as f64;
    let v1 = x.iter().map(|v| (v - m1).powi(2)).sum::<f64>() / (n1 - 1) as f64;
    let v2 = y.iter().map(|v| (v - m2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

    if equal_var {
        // Pooled variance
        let sp = ((n1 - 1) as f64 * v1 + (n2 - 1) as f64 * v2) / (n1 + n2 - 2) as f64;
        let se = (sp / n1 as f64 + sp / n2 as f64).sqrt();
        let t = (m1 - m2) / se;
        let df = (n1 + n2 - 2) as f64;
        let t_dist = TDist::new(df);
        let p = 2.0 * t_dist.sf(t.abs());
        (t, p)
    } else {
        // Welch's t-test
        let se = (v1 / n1 as f64 + v2 / n2 as f64).sqrt();
        let t = (m1 - m2) / se;
        // Welch-Satterthwaite degrees of freedom
        let num = (v1 / n1 as f64 + v2 / n2 as f64).powi(2);
        let den = (v1 / n1 as f64).powi(2) / (n1 - 1) as f64
            + (v2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let df = num / den;
        let t_dist = TDist::new(df);
        let p = 2.0 * t_dist.sf(t.abs());
        (t, p)
    }
}

/// Paired t-test. Returns (t-statistic, two-tailed p-value).
pub fn ttest_rel(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len());
    let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
    ttest_1samp(&diffs, 0.0)
}

/// One-way ANOVA F-test. Returns (F-statistic, p-value).
pub fn f_oneway(groups: &[Vec<f64>]) -> (f64, f64) {
    let k = groups.len();
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let grand_mean: f64 = groups.iter().flatten().sum::<f64>() / n_total as f64;

    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for group in groups {
        let n_i = group.len() as f64;
        let mean_i = group.iter().sum::<f64>() / n_i;
        ss_between += n_i * (mean_i - grand_mean).powi(2);
        ss_within += group.iter().map(|v| (v - mean_i).powi(2)).sum::<f64>();
    }

    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f = ms_between / ms_within;

    let f_dist = FDist::new(df_between, df_within);
    let p = f_dist.sf(f);
    (f, p)
}

/// Mann-Whitney U test (normal approximation for large samples).
pub fn mannwhitneyu(x: &[f64], y: &[f64]) -> Result<(f64, f64), String> {
    let n1 = x.len();
    let n2 = y.len();
    if n1 == 0 || n2 == 0 {
        return Err("Samples must not be empty".to_string());
    }

    // Compute U statistic
    let mut u1 = 0.0;
    for &xi in x {
        for &yj in y {
            if xi > yj {
                u1 += 1.0;
            } else if (xi - yj).abs() < 1e-15 {
                u1 += 0.5;
            }
        }
    }
    let u2 = n1 as f64 * n2 as f64 - u1;
    let u = u1.min(u2);

    // Normal approximation
    let mu = n1 as f64 * n2 as f64 / 2.0;
    let sigma = ((n1 as f64 * n2 as f64 * (n1 as f64 + n2 as f64 + 1.0)) / 12.0).sqrt();
    let z = (u - mu) / sigma;
    let p = 2.0 * Norm::cdf(-z.abs(), 0.0, 1.0);

    Ok((u, p))
}

/// Wilcoxon signed-rank test.
pub fn wilcoxon(x: &[f64]) -> Result<(f64, f64), String> {
    let n = x.len();
    if n < 5 {
        return Err("Need at least 5 observations".to_string());
    }

    // Compute signed ranks
    let mut abs_vals: Vec<(f64, f64)> = x
        .iter()
        .map(|&v| (v.abs(), v.signum()))
        .filter(|&(abs, _)| abs > 1e-15)
        .collect();
    abs_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n_eff = abs_vals.len();
    let mut ranks = vec![0.0; n_eff];
    let mut i = 0;
    while i < n_eff {
        let mut j = i;
        while j < n_eff && (abs_vals[j].0 - abs_vals[i].0).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + 1..=j).map(|r| r as f64).sum::<f64>() / (j - i) as f64;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    let w_pos: f64 = abs_vals
        .iter()
        .zip(ranks.iter())
        .filter(|((_, sign), _)| *sign > 0.0)
        .map(|(_, rank)| *rank)
        .sum();
    let w_neg: f64 = abs_vals
        .iter()
        .zip(ranks.iter())
        .filter(|((_, sign), _)| *sign < 0.0)
        .map(|(_, rank)| *rank)
        .sum();
    let w = w_pos.min(w_neg);

    // Normal approximation
    let mu = n_eff as f64 * (n_eff as f64 + 1.0) / 4.0;
    let sigma = (n_eff as f64 * (n_eff as f64 + 1.0) * (2.0 * n_eff as f64 + 1.0) / 24.0).sqrt();
    let z = (w - mu) / sigma;
    let p = 2.0 * Norm::cdf(-z.abs(), 0.0, 1.0);

    Ok((w, p))
}

/// Two-sample Kolmogorov-Smirnov test.
pub fn ks_2samp(x: &[f64], y: &[f64]) -> (f64, f64) {
    let mut all: Vec<(f64, i8)> = Vec::new();
    for &v in x {
        all.push((v, 0));
    }
    for &v in y {
        all.push((v, 1));
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n1 = x.len() as f64;
    let n2 = y.len() as f64;
    let mut d_max = 0.0;
    let mut cdf1 = 0.0;
    let mut cdf2 = 0.0;

    for &(_, group) in &all {
        if group == 0 {
            cdf1 += 1.0 / n1;
        } else {
            cdf2 += 1.0 / n2;
        }
        let d = (cdf1 - cdf2).abs();
        if d > d_max {
            d_max = d;
        }
    }

    // Asymptotic p-value using Kolmogorov distribution
    let n_eff = (n1 * n2) / (n1 + n2);
    let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * d_max;
    let mut p = 0.0;
    for k in 1..=100 {
        let term = (-2.0 * (k as f64) * (k as f64) * lambda * lambda).exp();
        if k % 2 == 1 {
            p += term;
        } else {
            p -= term;
        }
    }
    p = (2.0 * p).max(0.0).min(1.0);

    (d_max, p)
}

/// Shapiro-Wilk normality test (simplified approximation).
pub fn shapiro(x: &[f64]) -> Result<(f64, f64), String> {
    let n = x.len();
    if n < 3 || n > 5000 {
        return Err("Sample size must be between 3 and 5000".to_string());
    }

    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = sorted.iter().sum::<f64>() / n as f64;
    let ss: f64 = sorted.iter().map(|v| (v - mean).powi(2)).sum();

    if ss < 1e-15 {
        return Err("All values are equal".to_string());
    }

    // Compute a_i coefficients using normal order statistics (Royston's approximation)
    let mut a = vec![0.0; n];
    let n_f = n as f64;
    for i in 0..n {
        let p = (i as f64 + 1.0 - 0.375) / (n_f + 0.25);
        a[i] = norm_ppf(p);
    }
    // Normalize: sum of a_i^2 matters
    let a_sq_sum: f64 = a.iter().map(|v| v * v).sum();
    for ai in a.iter_mut() {
        *ai /= a_sq_sum.sqrt();
    }

    // W statistic
    let num: f64 = a.iter().zip(sorted.iter()).map(|(ai, xi)| ai * xi).sum::<f64>().powi(2);
    let w = num / ss;

    // Approximate p-value using Royston's method (logit transform)
    let mu = -1.2725 + 1.0521 * n_f.ln();
    let sigma = 1.0308 - 0.26758 * n_f.ln();
    let z = (-(1.0 - w).ln() - mu) / sigma;
    let p = 1.0 - Norm::cdf(z, 0.0, 1.0);

    Ok((w, p))
}

/// Chi-square test of independence of variables in a contingency table.
///
/// Returns (chi2_statistic, p_value, degrees_of_freedom, expected_frequencies).
pub fn chi2_contingency(observed: &Array2<f64>) -> Result<(f64, f64, usize, Array2<f64>), String> {
    let (nrows, ncols) = observed.dim();
    if nrows < 2 || ncols < 2 {
        return Err("Contingency table must have at least 2 rows and 2 columns".to_string());
    }

    for &val in observed.iter() {
        if val < 0.0 {
            return Err("All entries in observed table must be non-negative".to_string());
        }
    }

    let mut row_sums = vec![0.0; nrows];
    let mut col_sums = vec![0.0; ncols];
    let mut total = 0.0;

    for r in 0..nrows {
        for c in 0..ncols {
            let v = observed[[r, c]];
            row_sums[r] += v;
            col_sums[c] += v;
            total += v;
        }
    }

    if total == 0.0 {
        return Err("Total sum of table entries cannot be zero".to_string());
    }

    let mut expected = Array2::<f64>::zeros((nrows, ncols));
    let mut chi2 = 0.0;

    for r in 0..nrows {
        for c in 0..ncols {
            let exp = (row_sums[r] * col_sums[c]) / total;
            expected[[r, c]] = exp;
            if exp > 0.0 {
                let diff = observed[[r, c]] - exp;
                chi2 += diff * diff / exp;
            }
        }
    }

    let dof = (nrows - 1) * (ncols - 1);
    let chi2_dist = Chi2Dist::new(dof as f64);
    let p_value = (1.0 - chi2_dist.cdf(chi2)).clamp(0.0, 1.0);

    Ok((chi2, p_value, dof, expected))
}

/// Compute the first Wasserstein distance between two 1D empirical distributions.
pub fn wasserstein_distance(u_values: &Array1<f64>, v_values: &Array1<f64>) -> f64 {
    let n_u = u_values.len();
    let n_v = v_values.len();
    if n_u == 0 || n_v == 0 {
        return 0.0;
    }

    let mut u_sorted = u_values.to_vec();
    let mut v_sorted = v_values.to_vec();
    u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut all_points = Vec::with_capacity(n_u + n_v);
    all_points.extend_from_slice(&u_sorted);
    all_points.extend_from_slice(&v_sorted);
    all_points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_points.dedup();

    let mut total_dist = 0.0;
    let mut idx_u = 0;
    let mut idx_v = 0;

    for i in 0..all_points.len().saturating_sub(1) {
        let x_curr = all_points[i];
        let x_next = all_points[i + 1];
        let dx = x_next - x_curr;

        while idx_u < n_u && u_sorted[idx_u] <= x_curr {
            idx_u += 1;
        }
        while idx_v < n_v && v_sorted[idx_v] <= x_curr {
            idx_v += 1;
        }

        let cdf_u = idx_u as f64 / n_u as f64;
        let cdf_v = idx_v as f64 / n_v as f64;

        total_dist += (cdf_u - cdf_v).abs() * dx;
    }

    total_dist
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lgamma() {
        // lgamma(1) = 0, lgamma(2) = 0, lgamma(3) = ln(2)
        assert!((lgamma(1.0) - 0.0).abs() < 1e-10);
        assert!((lgamma(2.0) - 0.0).abs() < 1e-10);
        assert!((lgamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-10); // Γ(5)=24
    }

    #[test]
    fn test_gamma() {
        assert!((gamma(5.0) - 24.0).abs() < 1e-10);
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_betainc() {
        // I_0(a,b) = 0, I_1(a,b) = 1
        assert!((betainc(2.0, 3.0, 0.0) - 0.0).abs() < 1e-14);
        assert!((betainc(2.0, 3.0, 1.0) - 1.0).abs() < 1e-14);
        // Symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
        let a = 2.0;
        let b = 3.0;
        let x = 0.4;
        assert!((betainc(a, b, x) + betainc(b, a, 1.0 - x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gammainc() {
        // P(a, 0) = 0
        assert!((gammainc(1.0, 0.0) - 0.0).abs() < 1e-14);
        // P(1, x) = 1 - exp(-x)
        assert!((gammainc(1.0, 1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
        // P(a, inf) -> 1
        assert!((gammainc(2.0, 100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_ppf() {
        assert!((norm_ppf(0.5) - 0.0).abs() < 1e-6);
        assert!((norm_ppf(0.975) - 1.95996).abs() < 0.001);
        assert!((norm_ppf(0.025) + 1.95996).abs() < 0.001);
    }

    #[test]
    fn test_norm_dist() {
        let nd = NormDist::new(0.0, 1.0);
        // CDF at 0 should be 0.5
        let cdf0 = nd.cdf(0.0);
        assert!((cdf0 - 0.5).abs() < 1e-6, "cdf(0) = {}", cdf0);
        // ppf round-trip
        let p50 = nd.ppf(0.5);
        assert!(p50.abs() < 1e-4, "ppf(0.5) = {}", p50);
        let p975 = nd.ppf(0.975);
        assert!((p975 - 1.96).abs() < 0.01, "ppf(0.975) = {}", p975);
        assert!((nd.mean() - 0.0).abs() < 1e-14);
        assert!((nd.var() - 1.0).abs() < 1e-14);

        // PDF peak at mean
        assert!(nd.pdf(0.0) > nd.pdf(0.1));
        // Symmetry
        assert!((nd.pdf(-1.0) - nd.pdf(1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_t_dist() {
        let td = TDist::new(10.0);
        // Symmetry
        assert!((td.cdf(0.0) - 0.5).abs() < 1e-10);
        assert!((td.cdf(-2.0) + td.cdf(2.0) - 1.0).abs() < 1e-10);
        // ppf round-trip
        let p = 0.975;
        let t_val = td.ppf(p);
        assert!((td.cdf(t_val) - p).abs() < 1e-6);
    }

    #[test]
    fn test_chi2_dist() {
        let cd = Chi2Dist::new(1.0);
        // CDF at 0 is 0
        assert!((cd.cdf(0.0) - 0.0).abs() < 1e-14);
        // Mean = df
        assert!((cd.mean() - 1.0).abs() < 1e-14);
        // Var = 2*df
        assert!((cd.var() - 2.0).abs() < 1e-14);

        let cd2 = Chi2Dist::new(5.0);
        assert!((cd2.mean() - 5.0).abs() < 1e-14);
        // ppf round-trip
        let p = 0.95;
        let x = cd2.ppf(p);
        assert!((cd2.cdf(x) - p).abs() < 1e-5);
    }

    #[test]
    fn test_f_dist() {
        let fd = FDist::new(5.0, 10.0);
        assert!((fd.cdf(0.0) - 0.0).abs() < 1e-14);
        // Mean = dfd/(dfd-2) when dfd>2
        assert!((fd.mean() - 10.0 / 8.0).abs() < 1e-10);
        // ppf round-trip
        let p = 0.95;
        let x = fd.ppf(p);
        assert!((fd.cdf(x) - p).abs() < 1e-4);
    }

    #[test]
    fn test_beta_dist() {
        let bd = BetaDist::new(2.0, 5.0);
        assert!((bd.pdf(0.0) - 0.0).abs() < 1e-14);
        assert!((bd.pdf(1.0) - 0.0).abs() < 1e-14);
        assert!((bd.mean() - 2.0 / 7.0).abs() < 1e-10);
        // ppf round-trip
        let p = 0.5;
        let x = bd.ppf(p);
        assert!((bd.cdf(x) - p).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_dist() {
        let gd = GammaDist::new(2.0, 3.0);
        assert!((gd.mean() - 6.0).abs() < 1e-10);
        assert!((gd.var() - 18.0).abs() < 1e-10);
        // ppf round-trip
        let p = 0.75;
        let x = gd.ppf(p);
        assert!((gd.cdf(x) - p).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_dist() {
        let ed = ExponentialDist::new(2.0);
        assert!((ed.mean() - 2.0).abs() < 1e-10);
        assert!((ed.var() - 4.0).abs() < 1e-10);
        assert!((ed.cdf(0.0) - 0.0).abs() < 1e-14);
        // ppf is exact
        assert!((ed.ppf(0.5) + 2.0 * (0.5_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_dist() {
        let ud = UniformDist::new(0.0, 1.0);
        assert!((ud.mean() - 0.5).abs() < 1e-10);
        assert!((ud.var() - 1.0 / 12.0).abs() < 1e-10);
        assert!((ud.cdf(0.5) - 0.5).abs() < 1e-10);
        assert!((ud.ppf(0.5) - 0.5).abs() < 1e-10);
        assert!((ud.pdf(0.5) - 1.0).abs() < 1e-10);
        assert!((ud.pdf(1.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_dist() {
        let ld = LogNormalDist::new(0.0, 1.0);
        // Mean = exp(σ²/2) for μ=0
        assert!((ld.mean() - (0.5_f64).exp()).abs() < 1e-10);
        assert!((ld.cdf(0.0) - 0.0).abs() < 1e-14);
        // ppf round-trip
        let p = 0.75;
        let x = ld.ppf(p);
        let cdf_x = ld.cdf(x);
        assert!((cdf_x - p).abs() < 1e-4, "ppf({})={}, cdf({})={}", p, x, x, cdf_x);
    }

    #[test]
    fn test_bernoulli_dist() {
        let bd = BernoulliDist::new(0.7);
        assert!((bd.pmf(0) - 0.3).abs() < 1e-14);
        assert!((bd.pmf(1) - 0.7).abs() < 1e-14);
        assert!((bd.pmf(2) - 0.0).abs() < 1e-14);
        assert!((bd.mean() - 0.7).abs() < 1e-14);
        assert!((bd.var() - 0.21).abs() < 1e-14);
        assert!((bd.cdf(0) - 0.3).abs() < 1e-14);
        assert!((bd.cdf(1) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_bisect_ppf() {
        let nd = NormDist::new(0.0, 1.0);
        let result = bisect_ppf(|x| nd.cdf(x), 0.975, -5.0, 5.0, 1e-12, 200);
        assert!((result - 1.95996).abs() < 1e-5);
    }

    // ── Descriptive statistics tests ──

    #[test]
    fn test_kurtosis() {
        // Large sample from approximately uniform distribution
        let x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let k = kurtosis(&x, true);
        // Uniform distribution excess kurtosis is -1.2
        assert!(k.abs() < 1.5, "kurtosis = {}", k);
    }

    #[test]
    fn test_skew() {
        // Symmetric data: skew ~0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = skew(&x);
        assert!(s.abs() < 1e-10);

        // Right-skewed
        let y = vec![1.0, 1.0, 1.0, 1.0, 10.0];
        let s2 = skew(&y);
        assert!(s2 > 0.0);
    }

    #[test]
    fn test_describe() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (n, min, max, mean, var, _, _) = describe(&x);
        assert_eq!(n, 5);
        assert!((min - 1.0).abs() < 1e-14);
        assert!((max - 5.0).abs() < 1e-14);
        assert!((mean - 3.0).abs() < 1e-10);
        assert!((var - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_mode() {
        let x = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let (m, c) = mode(&x);
        assert!((m - 3.0).abs() < 1e-10);
        assert_eq!(c, 3);
    }

    #[test]
    fn test_iqr() {
        let x: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let q = iqr(&x);
        // Q1=25.5, Q3=75.5 for 1..=100
        assert!((q - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_sem() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let se = sem(&x);
        // s = sqrt(2.5), sem = s/sqrt(5)
        let expected = (2.5_f64).sqrt() / (5.0_f64).sqrt();
        assert!((se - expected).abs() < 1e-10);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution: entropy = ln(n)
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = entropy(&p);
        assert!((h - 4.0_f64.ln()).abs() < 1e-10);

        // Certain event: entropy = 0
        let p2 = vec![1.0, 0.0, 0.0];
        assert!((entropy(&p2) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_linregress() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let (slope, intercept, r, p, se) = linregress(&x, &y).unwrap();
        assert!((slope - 0.6).abs() < 1e-10, "slope = {}", slope);
        assert!((intercept - 2.2).abs() < 1e-10, "intercept = {}", intercept);
        // Manual: ss_xy=6.0, ss_xx=10.0, ss_yy=6.0, r=6/sqrt(60)=0.7746
        let expected_r = 6.0_f64 / 60.0_f64.sqrt();
        assert!((r - expected_r).abs() < 1e-10, "r = {}", r);
        assert!(se > 0.0);
        assert!(p > 0.0 && p < 1.0);
    }

    // ── Hypothesis test tests ──

    #[test]
    fn test_ttest_1samp() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (t, p) = ttest_1samp(&x, 3.0);
        // Mean is 3.0, so t should be ~0
        assert!(t.abs() < 1e-10);
        assert!((p - 1.0).abs() < 1e-6);

        let (t2, p2) = ttest_1samp(&x, 0.0);
        assert!(t2 > 0.0);
        assert!(p2 < 0.05);
    }

    #[test]
    fn test_ttest_ind() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let (t, p) = ttest_ind(&x, &y, true);
        assert!(t < 0.0);
        assert!(p < 0.05);

        // Welch's
        let (t2, p2) = ttest_ind(&x, &y, false);
        assert!(t2 < 0.0);
        assert!(p2 < 0.05);
    }

    #[test]
    fn test_ttest_rel() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let (_t, _p) = ttest_rel(&x, &y);
        // Constant difference, so t should be -inf or very large
        // Actually diff = -0.5 for all, mean = -0.5, std = 0 -> t = -inf
        // Let's use different values
        let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y2 = vec![2.0, 3.0, 5.0, 6.0, 8.0];
        let (t2, p2) = ttest_rel(&x2, &y2);
        assert!(t2 < 0.0);
        assert!(p2 < 0.05);
    }

    #[test]
    fn test_f_oneway() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let g3 = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let (f, p) = f_oneway(&[g1, g2, g3]);
        assert!(f > 0.0);
        assert!(p < 0.01); // very different groups
    }

    #[test]
    fn test_mannwhitneyu() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let (u, p) = mannwhitneyu(&x, &y).unwrap();
        assert!(u == 0.0); // no overlaps
        assert!(p < 0.05);
    }

    #[test]
    fn test_wilcoxon() {
        // Use data with both positive and negative values so W_pos and W_neg are both > 0
        let x = vec![-3.0, -1.0, 0.5, 2.0, 4.0, -2.0, 1.0, 3.0];
        let (w, p) = wilcoxon(&x).unwrap();
        assert!(w > 0.0, "w = {}", w);
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn test_ks_2samp() {
        // Same distribution
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let (d, p) = ks_2samp(&x, &y);
        assert!(d < 0.5);
        assert!(p > 0.01);

        // Very different distributions
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (d2, p2) = ks_2samp(&a, &b);
        assert!(d2 > 0.8);
        assert!(p2 < 0.05);
    }

    #[test]
    fn test_shapiro() {
        // Approximately normal data
        let x: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();
        let (w, p) = shapiro(&x).unwrap();
        assert!(w > 0.0 && w <= 1.0);
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn test_chi2_contingency() {
        // Example: 2x2 table [[10, 20], [20, 40]] has independent rows and cols -> chi2 should be 0
        let obs_indep = Array2::from_shape_vec((2, 2), vec![10.0, 20.0, 20.0, 40.0]).unwrap();
        let (chi2, p_val, dof, _) = chi2_contingency(&obs_indep).unwrap();
        assert_eq!(dof, 1);
        assert!(chi2 < 1e-10);
        assert!((p_val - 1.0).abs() < 1e-6);

        // Dependent table: [[10, 10], [20, 10]]
        let obs = Array2::from_shape_vec((2, 2), vec![10.0, 10.0, 20.0, 10.0]).unwrap();
        let (chi2_dep, _, dof_dep, expected) = chi2_contingency(&obs).unwrap();
        assert_eq!(dof_dep, 1);
        assert!(chi2_dep > 0.0);
        assert_eq!(expected.dim(), (2, 2));
    }

    #[test]
    fn test_wasserstein_distance() {
        // Identical distributions -> distance = 0
        let u = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let dist_zero = wasserstein_distance(&u, &u);
        assert!(dist_zero.abs() < 1e-12);

        // Shifted distribution: distance should equal the shift
        let v = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let dist = wasserstein_distance(&u, &v);
        assert!((dist - 1.0).abs() < 1e-6);
    }
}
