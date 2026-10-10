use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Gamma family
// ---------------------------------------------------------------------------

/// Gamma function Γ(x).
///
/// Uses Lanczos approximation for x > 0 and the reflection formula for x < 0.
/// Returns `f64::NAN` for non-positive integers (poles).
pub fn gamma(x: f64) -> f64 {
    if x > 0.0 {
        gammaln(x).exp()
    } else if x == 0.0 {
        f64::INFINITY
    } else if x.fract() == 0.0 && x == x.trunc() {
        // Negative integer → pole
        f64::NAN
    } else {
        // Reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        PI / ((PI * x).sin() * gamma(1.0 - x))
    }
}

/// Log of the absolute value of the Gamma function, ln|Γ(x)|.
///
/// Uses the Lanczos approximation (g=7, n=9 coefficients).
pub fn gammaln(x: f64) -> f64 {
    if x < 0.5 {
        // Use reflection: ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln|Γ(1-x)|
        return PI.ln() - (PI * x).sin().abs().ln() - gammaln(1.0 - x);
    }

    let x = x - 1.0;
    // Lanczos coefficients (g=7, n=9)
    let c = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let mut y = c[0];
    for i in 1..9 {
        y += c[i] / (x + i as f64);
    }

    let t = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + y.ln()
}

/// Digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the recurrence relation to shift x into the asymptotic region,
/// then applies the series expansion.
pub fn digamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0;

    // Use recurrence to shift x > 8
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion for large x
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    result += x.ln() - 0.5 * inv_x
        - inv_x2 / 12.0 * (1.0
            - inv_x2 / 10.0 * (1.0 - inv_x2 / 42.0 * (1.0 - inv_x2 / 40.0)));

    result
}

/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
pub fn beta(a: f64, b: f64) -> f64 {
    betaln(a, b).exp()
}

/// Log of the Beta function, ln B(a, b).
pub fn betaln(a: f64, b: f64) -> f64 {
    gammaln(a) + gammaln(b) - gammaln(a + b)
}

// ---------------------------------------------------------------------------
// Error functions
// ---------------------------------------------------------------------------

/// Scaled complementary error function erfcx(x) = exp(x²) * erfc(x).
///
/// Uses a rational approximation for numerical stability.
pub fn erfcx(x: f64) -> f64 {
    if x < 0.0 {
        // erfcx(-x) = 2*exp(x²) - erfcx(x)
        let pos = erfcx(-x);
        2.0 * (x * x).exp() - pos
    } else if x < 6.0 {
        // Rational approximation (Abramowitz & Stegun 7.1.26 adapted)
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t
            * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erfc_val = poly * (-x * x).exp();
        erfc_val * (x * x).exp()
    } else {
        // For large x, use asymptotic: erfcx(x) ≈ 1/(sqrt(π)*x)
        let inv_sqrt_pi = 1.0 / PI.sqrt();
        inv_sqrt_pi / x * (1.0 - 0.5 / (x * x) + 0.75 / (x * x * x * x))
    }
}

/// Inverse error function.
///
/// Returns x such that erf(x) = p, for -1 < p < 1.
/// Uses a rational approximation.
pub fn erfinv(p: f64) -> f64 {
    if p == 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p <= -1.0 {
        return f64::NEG_INFINITY;
    }

    let sign = if p < 0.0 { -1.0 } else { 1.0 };
    let p = p.abs();

    // Initial guess using rational approximation
    // Based on the approximation from Giles (2010)
    let a = 0.147;
    let ln1p2 = (1.0 - p * p).ln();
    let tt1 = 2.0 / (std::f64::consts::PI * a) + ln1p2 / 2.0;
    let tt2 = ln1p2 / a;
    let mut x = sign * (-tt1 + (tt1 * tt1 - tt2).sqrt()).sqrt();

    // Newton-Raphson refinement: solve erf(x) = p
    // x_{n+1} = x_n - (erf(x_n) - p) / (2/sqrt(pi) * exp(-x_n^2))
    let sqrt_pi = std::f64::consts::PI.sqrt();
    for _ in 0..5 {
        let erf_x = erf(x);
        let deriv = (2.0 / sqrt_pi) * (-x * x).exp();
        x -= (erf_x - p) / deriv;
    }

    sign * x
}

/// Error function using Horner form rational approximation.
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Abramowitz & Stegun 7.1.26 (max error 1.5e-7)
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = 0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3
        - 1.453152027 * t4
        + 1.061405429 * t5;

    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal cumulative distribution function (CDF).
pub fn ndtr(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse of the standard normal CDF.
pub fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    std::f64::consts::SQRT_2 * erfinv(2.0 * p - 1.0)
}

/// Alias for `ndtri`.
pub fn norminv(p: f64) -> f64 {
    ndtri(p)
}

// ---------------------------------------------------------------------------
// Logistic functions
// ---------------------------------------------------------------------------

/// Logistic sigmoid (expit) function: 1 / (1 + exp(-x)).
///
/// Numerically stable for all inputs.
pub fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Logit function: log(p / (1 - p)).
///
/// Returns `f64::NAN` for p outside (0, 1).
pub fn logit(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        f64::NAN
    } else {
        (p / (1.0 - p)).ln()
    }
}

// ---------------------------------------------------------------------------
// Numerically stable operations
// ---------------------------------------------------------------------------

/// Log of sum of exponentials: log(sum(exp(x_i))).
///
/// Computed in a numerically stable way by subtracting the max value first.
pub fn logsumexp(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::INFINITY {
        return f64::INFINITY;
    }
    let sum: f64 = x.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Numerically stable softmax: softmax(x)_i = exp(x_i) / sum(exp(x_j)).
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Bessel functions
// ---------------------------------------------------------------------------

/// Modified Bessel function of the first kind I₀(x).
///
/// Uses the Chebyshev approximation from Numerical Recipes.
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y
            * (3.5156229
                + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Modified Bessel function of the first kind I₁(x).
///
/// Uses the Chebyshev approximation from Numerical Recipes.
pub fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        sign * ax / 2.0
            * (0.5
                + y * (0.87890594
                    + y * (0.51498869 + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))))
    } else {
        let y = 3.75 / ax;
        sign * (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (-0.03988024
                    + y * (-0.00362018
                        + y * (0.00163801
                            + y * (-0.01031555
                                + y * (0.02282967
                                    + y * (-0.02895312 + y * (0.01787654 + y * (-0.00420059)))))))))
    }
}

/// Bessel function of the first kind J₀(x).
///
/// Uses a rational approximation for different ranges.
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        // Rational approximation (Abramowitz & Stegun 9.4.1)
        let num = 57568490574.0
            + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let den = 57568490411.0
            + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;

        let f0 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let g0 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * 0.7621095161e-6));

        (2.0 / PI / ax).sqrt() * (f0 * xx.cos() - z * g0 * xx.sin())
    }
}

/// Bessel function of the first kind J₁(x).
///
/// Uses a rational approximation for different ranges.
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72362614232.0
                + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let den = 144725228442.0
            + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;

        let f1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let g1 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6)));

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        sign * (2.0 / PI / ax).sqrt() * (f1 * xx.cos() - z * g1 * xx.sin())
    }
}

// ---------------------------------------------------------------------------
// Combinatorial
// ---------------------------------------------------------------------------

/// Factorial n! as f64.
///
/// Handles n up to 170 (the f64 overflow limit). Returns `f64::INFINITY` for n > 170.
pub fn factorial(n: u64) -> f64 {
    if n > 170 {
        return f64::INFINITY;
    }
    let mut result = 1.0_f64;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// Log of factorial: ln(n!).
///
/// Uses `gammaln` for numerical stability.
pub fn factorialln(n: u64) -> f64 {
    gammaln(n as f64 + 1.0)
}

/// Binomial coefficient C(n, k) as f64.
pub fn comb(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k = if k > n - k { n - k } else { k };
    if n <= 60 {
        let mut res: u128 = 1;
        for i in 1..=k {
            res = res * ((n - k + i) as u128) / (i as u128);
        }
        res as f64
    } else {
        (factorialln(n) - factorialln(k) - factorialln(n - k)).exp().round()
    }
}

/// Permutation P(n, k) = n! / (n - k)! as f64.
pub fn perm(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    if n <= 20 {
        let mut res: u128 = 1;
        for i in 0..k {
            res *= (n - i) as u128;
        }
        res as f64
    } else {
        (factorialln(n) - factorialln(n - k)).exp().round()
    }
}

/// Pochhammer symbol (rising factorial) (x)_n = Γ(x + n) / Γ(x).
pub fn poch(x: f64, n: f64) -> f64 {
    if n == 0.0 {
        return 1.0;
    }
    (gammaln(x + n) - gammaln(x)).exp()
}

// ---------------------------------------------------------------------------
// Bessel, Hypergeometric, Elliptic, and Wave functions
// ---------------------------------------------------------------------------

/// Bessel function of the first kind of real order `v` and argument `x`: $J_v(x)$.
pub fn jv(v: f64, x: f64) -> f64 {
    if v == 0.0 {
        return bessel_j0(x);
    }
    if v == 1.0 {
        return bessel_j1(x);
    }
    if x == 0.0 {
        return if v == 0.0 { 1.0 } else { 0.0 };
    }
    if x < 0.0 {
        let n = v.round();
        if (v - n).abs() < 1e-12 {
            let sign = if (n as i64) % 2 == 0 { 1.0 } else { -1.0 };
            return sign * jv(v, -x);
        } else {
            return f64::NAN;
        }
    }

    if x > 30.0 {
        // Asymptotic expansion
        let theta = x - 0.5 * v * PI - 0.25 * PI;
        return (2.0 / (PI * x)).sqrt() * theta.cos();
    }

    let z = 0.5 * x;
    let mut sum = 0.0;
    let mut term = z.powf(v) / gamma(v + 1.0);
    sum += term;

    for m in 1..80 {
        term *= -z * z / (m as f64 * (m as f64 + v));
        sum += term;
        if term.abs() < 1e-15 * sum.abs().max(1e-15) {
            break;
        }
    }
    sum
}

/// Modified Bessel function of the first kind of real order `v` and argument `x`: $I_v(x)$.
pub fn iv(v: f64, x: f64) -> f64 {
    if v == 0.0 {
        return bessel_i0(x);
    }
    if v == 1.0 {
        return bessel_i1(x);
    }
    if x == 0.0 {
        return if v == 0.0 { 1.0 } else { 0.0 };
    }
    let sign = if x < 0.0 {
        let n = v.round();
        if (v - n).abs() < 1e-12 {
            if (n as i64) % 2 == 0 { 1.0 } else { -1.0 }
        } else {
            return f64::NAN;
        }
    } else {
        1.0
    };
    let x = x.abs();

    if x > 35.0 {
        return (x.exp() / (2.0 * PI * x).sqrt()) * (1.0 - (4.0 * v * v - 1.0) / (8.0 * x));
    }

    let z = 0.5 * x;
    let mut sum = 0.0;
    let mut term = z.powf(v) / gamma(v + 1.0);
    sum += term;

    for m in 1..80 {
        term *= z * z / (m as f64 * (m as f64 + v));
        sum += term;
        if term.abs() < 1e-15 * sum.abs().max(1e-15) {
            break;
        }
    }
    sign * sum
}

/// Bessel function of the second kind of real order `v`: $Y_v(x)$.
pub fn yv(v: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let n = v.round();
    let effective_v = if (v - n).abs() < 1e-5 {
        n + 1e-6
    } else {
        v
    };

    let sin_v_pi = (effective_v * PI).sin();
    let cos_v_pi = (effective_v * PI).cos();
    (jv(effective_v, x) * cos_v_pi - jv(-effective_v, x)) / sin_v_pi
}

/// Modified Bessel function of the second kind of real order `v`: $K_v(x)$.
pub fn kv(v: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let n = v.round();
    let effective_v = if (v - n).abs() < 1e-5 {
        n + 1e-6
    } else {
        v
    };

    let sin_v_pi = (effective_v * PI).sin();
    0.5 * PI * (iv(-effective_v, x) - iv(effective_v, x)) / sin_v_pi
}

/// Confluent hypergeometric function $_1F_1(a; b; x)$.
pub fn hyp1f1(a: f64, b: f64, x: f64) -> f64 {
    if b <= 0.0 && b == b.round() {
        return f64::NAN;
    }
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 0..500 {
        term *= ((a + k as f64) / (b + k as f64)) * (x / (k + 1) as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum
}

/// Gauss hypergeometric function $_2F_1(a, b; c; x)$.
pub fn hyp2f1(a: f64, b: f64, c: f64, x: f64) -> f64 {
    if c <= 0.0 && c == c.round() {
        return f64::NAN;
    }
    assert!(x.abs() < 1.0 || (x.abs() - 1.0).abs() < 1e-12, "x must be in [-1, 1]");
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 0..500 {
        term *= ((a + k as f64) * (b + k as f64) / (c + k as f64)) * (x / (k + 1) as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum
}

/// Complete elliptic integral of the first kind $K(m) = \int_0^{\pi/2} (1 - m \sin^2 \theta)^{-1/2} d\theta$.
pub fn ellipk(m: f64) -> f64 {
    if m < 0.0 || m >= 1.0 {
        return f64::NAN;
    }
    let mut a = 1.0;
    let mut g = (1.0 - m).sqrt();

    while (a - g).abs() > 1e-15 {
        let a_next = 0.5 * (a + g);
        let g_next = (a * g).sqrt();
        a = a_next;
        g = g_next;
    }

    PI / (2.0 * a)
}

/// Complete elliptic integral of the second kind $E(m) = \int_0^{\pi/2} (1 - m \sin^2 \theta)^{1/2} d\theta$.
pub fn ellipe(m: f64) -> f64 {
    if m < 0.0 || m > 1.0 {
        return f64::NAN;
    }
    if m == 1.0 {
        return 1.0;
    }
    let mut a = 1.0;
    let mut g = (1.0 - m).sqrt();
    let mut power = 1.0;
    let mut sum = 0.5 * m;

    while (a - g).abs() > 1e-15 {
        let c = 0.5 * (a - g);
        let a_next = 0.5 * (a + g);
        let g_next = (a * g).sqrt();
        power *= 2.0;
        sum += power * c * c;
        a = a_next;
        g = g_next;
    }

    (PI / (2.0 * a)) * (1.0 - sum)
}

/// Airy functions $Ai(x)$, $Ai'(x)$, $Bi(x)$, $Bi'(x)$.
pub fn airy(x: f64) -> (f64, f64, f64, f64) {
    let ai0 = 0.355_028_053_887_817_2;
    let aip0 = -0.258_819_403_792_806_9;
    let sqrt3 = 3.0_f64.sqrt();

    if x.abs() < 1e-15 {
        return (ai0, aip0, sqrt3 * ai0, -sqrt3 * aip0);
    }

    // Taylor series around 0
    let mut f = 1.0;
    let mut g = x;
    let mut fp = 0.0;
    let mut gp = 1.0;

    let mut term_f = 1.0;
    let mut term_g = x;

    for k in 1..40 {
        let k3 = 3 * k;
        let den_f = ((k3 - 1) * k3) as f64;
        term_f *= (x * x * x) / den_f;
        f += term_f;
        fp += (k3 as f64 / x) * term_f;

        let den_g = (k3 * (k3 + 1)) as f64;
        term_g *= (x * x * x) / den_g;
        g += term_g;
        gp += ((k3 + 1) as f64 / x) * term_g;

        if term_f.abs() < 1e-16 && term_g.abs() < 1e-16 {
            break;
        }
    }

    let ai = ai0 * f + aip0 * g;
    let bi = sqrt3 * (ai0 * f - aip0 * g);
    let aip = ai0 * fp + aip0 * gp;
    let bip = sqrt3 * (ai0 * fp - aip0 * gp);

    (ai, aip, bi, bip)
}

/// Spherical harmonic $Y_n^m(\theta, \phi)$ returning `(real, imag)`.
///
/// * `m`: Order of harmonic, $-n \le m \le n$.
/// * `n`: Degree of harmonic, $n \ge 0$.
/// * `theta`: Polar / colatitude angle $[0, \pi]$.
/// * `phi`: Azimuthal angle $[0, 2\pi)$.
pub fn sph_harm(m: i64, n: i64, theta: f64, phi: f64) -> (f64, f64) {
    assert!(n >= 0 && m.abs() <= n, "Invalid n or m");
    let x = theta.cos();
    let abs_m = m.unsigned_abs() as usize;
    let n_u = n as usize;

    // Associated Legendre polynomial P_n^{|m|}(x)
    let mut pmm = 1.0;
    if abs_m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0;
        for _ in 1..=abs_m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    let pnm = if n_u == abs_m {
        pmm
    } else {
        let mut pmmp1 = x * (2 * abs_m + 1) as f64 * pmm;
        if n_u == abs_m + 1 {
            pmmp1
        } else {
            let mut pll = 0.0;
            for ll in (abs_m + 2)..=n_u {
                pll = ((2 * ll - 1) as f64 * x * pmmp1 - (ll + abs_m - 1) as f64 * pmm)
                    / (ll - abs_m) as f64;
                pmm = pmmp1;
                pmmp1 = pll;
            }
            pll
        }
    };

    let p_eff = if m < 0 {
        let sign = if (-m) % 2 == 0 { 1.0 } else { -1.0 };
        sign * (factorial((n - abs_m as i64) as u64) / factorial((n + abs_m as i64) as u64)) * pnm
    } else {
        pnm
    };

    let norm = (((2 * n + 1) as f64 / (4.0 * PI))
        * (factorial((n - m) as u64) / factorial((n + m) as u64)))
        .sqrt();

    let magnitude = norm * p_eff;
    let angle = m as f64 * phi;
    (magnitude * angle.cos(), magnitude * angle.sin())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Gamma family ---

    #[test]
    fn test_gamma_positive_integers() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-9);
        assert!((gamma(2.0) - 1.0).abs() < 1e-9);
        assert!((gamma(3.0) - 2.0).abs() < 1e-9);
        assert!((gamma(4.0) - 6.0).abs() < 1e-9);
        assert!((gamma(5.0) - 24.0).abs() < 1e-9);
        assert!((gamma(6.0) - 120.0).abs() < 1e-9);
    }

    #[test]
    fn test_gamma_half_integer() {
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-9);
        assert!((gamma(1.5) - 0.5 * PI.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_gamma_negative_half() {
        // Γ(-0.5) = -2√π
        assert!((gamma(-0.5) - (-2.0 * PI.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn test_gammaln() {
        assert!((gammaln(5.0) - 24.0_f64.ln()).abs() < 1e-9);
        assert!((gammaln(1.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_digamma() {
        // ψ(1) = -γ (Euler-Mascheroni constant ≈ -0.5772)
        let euler_mascheroni = 0.5772156649015329;
        assert!((digamma(1.0) + euler_mascheroni).abs() < 1e-4);
        // ψ(2) = 1 - γ
        assert!((digamma(2.0) - (1.0 - euler_mascheroni)).abs() < 1e-4);
    }

    #[test]
    fn test_beta() {
        // B(2, 3) = 1/12
        assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-9);
        // B(1, 1) = 1
        assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_betaln() {
        assert!((betaln(2.0, 3.0) - (1.0 / 12.0_f64).ln()).abs() < 1e-9);
    }

    // --- Error functions ---

    #[test]
    fn test_erfcx_zero() {
        // erfcx(0) = erfc(0) * exp(0) = 1
        assert!((erfcx(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_erfinv_zero() {
        assert!((erfinv(0.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_erfinv_half() {
        // erfinv(0.5) ≈ 0.4769362762044699
        assert!((erfinv(0.5) - 0.4769362762044699).abs() < 1e-4);
    }

    #[test]
    fn test_erfinv_symmetry() {
        // erfinv is odd: erfinv(-x) = -erfinv(x)
        let x = 0.3;
        assert!((erfinv(-x) + erfinv(x)).abs() < 1e-9);
    }

    // --- Logistic functions ---

    #[test]
    fn test_expit_zero() {
        assert!((expit(0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_expit_large_positive() {
        assert!((expit(100.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_expit_large_negative() {
        assert!(expit(-100.0).abs() < 1e-9);
    }

    #[test]
    fn test_logit_half() {
        assert!((logit(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_logit_expit_roundtrip() {
        let x = 2.5;
        assert!((logit(expit(x)) - x).abs() < 1e-9);
    }

    // --- Numerically stable operations ---

    #[test]
    fn test_logsumexp_basic() {
        let x = [1.0, 2.0, 3.0];
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((logsumexp(&x) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_logsumexp_single() {
        let x = [5.0];
        assert!((logsumexp(&x) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let s = softmax(&x);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_uniform() {
        let x = [0.0, 0.0, 0.0];
        let s = softmax(&x);
        for &v in &s {
            assert!((v - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    // --- Bessel functions ---

    #[test]
    fn test_bessel_i0_zero() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_i1_zero() {
        assert!((bessel_i1(0.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_bessel_j0_zero() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_j1_zero() {
        assert!((bessel_j1(0.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_bessel_j0_first_zero() {
        // First zero of J₀ is at x ≈ 2.4048255577
        let x0 = 2.4048255577;
        assert!(bessel_j0(x0).abs() < 1e-4);
    }

    // --- Combinatorial ---

    #[test]
    fn test_factorial_basic() {
        assert!((factorial(0) - 1.0).abs() < 1e-9);
        assert!((factorial(1) - 1.0).abs() < 1e-9);
        assert!((factorial(5) - 120.0).abs() < 1e-9);
        assert!((factorial(10) - 3628800.0).abs() < 1e-9);
    }

    #[test]
    fn test_factorial_large() {
        // 170! is the largest factorial representable in f64
        let f170 = factorial(170);
        assert!(f170.is_finite());
        assert!(factorial(171).is_infinite());
    }

    #[test]
    fn test_factorialln() {
        assert!((factorialln(5) - 120.0_f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn test_comb() {
        assert!((comb(10, 3) - 120.0).abs() < 1e-6);
        assert!((comb(10, 0) - 1.0).abs() < 1e-9);
        assert!((comb(10, 10) - 1.0).abs() < 1e-9);
        assert!((comb(5, 3) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_perm() {
        assert!((perm(10, 3) - 720.0).abs() < 1e-6);
        assert!((perm(5, 2) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_poch() {
        // (x)_0 = 1
        assert!((poch(3.0, 0.0) - 1.0).abs() < 1e-9);
        // (2)_3 = 2*3*4 = 24
        assert!((poch(2.0, 3.0) - 24.0).abs() < 1e-6);
        // (1)_n = n!
        assert!((poch(1.0, 5.0) - 120.0).abs() < 1e-6);
    }

    #[test]
    fn test_ndtr_and_ndtri() {
        // ndtr(0) = 0.5, ndtri(0.5) = 0
        assert!((ndtr(0.0) - 0.5).abs() < 1e-6);
        assert!(ndtri(0.5).abs() < 1e-6);

        let p = 0.841344746; // ~ 1 std dev
        let z = ndtri(p);
        assert!((z - 1.0).abs() < 1e-4);
        assert!((ndtr(1.0) - p).abs() < 1e-4);
    }

    #[test]
    fn test_bessel_arbitrary_order() {
        // J_0(0) = 1, J_1(0) = 0
        assert!((jv(0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!(jv(1.0, 0.0).abs() < 1e-6);
        assert!((jv(2.0, 1.0) - 0.1149034849).abs() < 1e-4);

        // I_0(0) = 1, I_1(0) = 0
        assert!((iv(0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((iv(2.0, 1.0) - 0.1357476697).abs() < 1e-4);

        // Y_0(1.0) ≈ 0.088256964
        assert!((yv(0.0, 1.0) - 0.088256964).abs() < 1e-3);

        // K_0(1.0) ≈ 0.421024438
        assert!((kv(0.0, 1.0) - 0.421024438).abs() < 1e-3);
    }

    #[test]
    fn test_hypergeometric() {
        // hyp1f1(1, 1, x) = e^x
        assert!((hyp1f1(1.0, 1.0, 1.0) - std::f64::consts::E).abs() < 1e-6);
        // hyp2f1(1, 1, 2, x) = -ln(1 - x) / x
        let x: f64 = 0.5;
        let expected = -(1.0 - x).ln() / x;
        assert!((hyp2f1(1.0, 1.0, 2.0, x) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_elliptic_integrals() {
        // K(0) = pi / 2
        assert!((ellipk(0.0) - PI / 2.0).abs() < 1e-9);
        // E(0) = pi / 2, E(1) = 1
        assert!((ellipe(0.0) - PI / 2.0).abs() < 1e-9);
        assert!((ellipe(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_airy_and_sph_harm() {
        let (ai0, aip0, bi0, _bip0) = airy(0.0);
        assert!((ai0 - 0.35502805).abs() < 1e-5);
        assert!((aip0 - (-0.2588194)).abs() < 1e-5);
        assert!((bi0 - 0.6149266).abs() < 1e-5);

        // Y_0^0(theta, phi) = 1 / sqrt(4 * pi)
        let (re, im) = sph_harm(0, 0, PI / 3.0, PI / 4.0);
        let expected = 1.0 / (4.0 * PI).sqrt();
        assert!((re - expected).abs() < 1e-6);
        assert!(im.abs() < 1e-9);
    }
}

