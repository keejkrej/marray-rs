use ndarray::{Array1, Array2};

/// 1D numerical integration using adaptive quadrature.
///
/// Computes a definite integral of `f` over `[a, b]` using adaptive Simpson's quadrature
/// with local error control.
///
/// Returns `(integral, estimated_abs_error)`.
pub fn quad<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    epsabs: Option<f64>,
    epsrel: Option<f64>,
) -> Result<(f64, f64), String> {
    if a == b {
        return Ok((0.0, 0.0));
    }

    let atol = epsabs.unwrap_or(1.49e-8);
    let rtol = epsrel.unwrap_or(1.49e-8);

    let (sign, a_val, b_val) = if a > b {
        (-1.0, b, a)
    } else {
        (1.0, a, b)
    };

    struct StackItem {
        a: f64,
        b: f64,
        fa: f64,
        fb: f64,
        fm: f64,
        s: f64,
        tol: f64,
        depth: usize,
    }

    let fa = f(a_val);
    let fb = f(b_val);
    let m = 0.5 * (a_val + b_val);
    let fm = f(m);
    let s0 = (b_val - a_val) / 6.0 * (fa + 4.0 * fm + fb);

    let mut stack = vec![StackItem {
        a: a_val,
        b: b_val,
        fa,
        fb,
        fm,
        s: s0,
        tol: atol,
        depth: 0,
    }];

    let mut total_integral = 0.0;
    let mut total_err = 0.0;
    let max_depth = 50;

    while let Some(item) = stack.pop() {
        let m = 0.5 * (item.a + item.b);
        let h = item.b - item.a;

        let m_left = 0.5 * (item.a + m);
        let m_right = 0.5 * (m + item.b);

        let fm_left = f(m_left);
        let fm_right = f(m_right);

        let s_left = (h / 12.0) * (item.fa + 4.0 * fm_left + item.fm);
        let s_right = (h / 12.0) * (item.fm + 4.0 * fm_right + item.fb);
        let s2 = s_left + s_right;

        let err = (s2 - item.s).abs() / 15.0;
        let local_tol = item.tol.max(rtol * s2.abs());

        if err <= local_tol || item.depth >= max_depth {
            total_integral += s2 + (s2 - item.s) / 15.0;
            total_err += err;
        } else {
            stack.push(StackItem {
                a: m,
                b: item.b,
                fa: item.fm,
                fb: item.fb,
                fm: fm_right,
                s: s_right,
                tol: 0.5 * item.tol,
                depth: item.depth + 1,
            });
            stack.push(StackItem {
                a: item.a,
                b: m,
                fa: item.fa,
                fb: item.fm,
                fm: fm_left,
                s: s_left,
                tol: 0.5 * item.tol,
                depth: item.depth + 1,
            });
        }
    }

    Ok((sign * total_integral, total_err))
}

/// 2D numerical integration over general limits.
///
/// Computes $\int_a^b \int_{g(x)}^{h(x)} f(y, x) dy dx$.
pub fn dblquad<F, G, H>(
    func: F,
    a: f64,
    b: f64,
    gfun: G,
    hfun: H,
    epsabs: Option<f64>,
    epsrel: Option<f64>,
) -> Result<(f64, f64), String>
where
    F: Fn(f64, f64) -> f64,
    G: Fn(f64) -> f64,
    H: Fn(f64) -> f64,
{
    let inner_integral = |x: f64| -> f64 {
        let y_min = gfun(x);
        let y_max = hfun(x);
        let integrand = |y: f64| func(y, x);
        quad(integrand, y_min, y_max, epsabs, epsrel)
            .map(|(val, _)| val)
            .unwrap_or(0.0)
    };

    quad(inner_integral, a, b, epsabs, epsrel)
}

/// Trapezoidal rule integration along an array.
pub fn trapz(y: &Array1<f64>, x: Option<&Array1<f64>>, dx: Option<f64>) -> f64 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }

    if let Some(x_arr) = x {
        assert_eq!(x_arr.len(), n, "x and y must have the same length");
        let mut sum = 0.0;
        for i in 0..n - 1 {
            let h = x_arr[i + 1] - x_arr[i];
            sum += 0.5 * h * (y[i] + y[i + 1]);
        }
        sum
    } else {
        let step = dx.unwrap_or(1.0);
        let mut sum = 0.0;
        for i in 0..n - 1 {
            sum += 0.5 * step * (y[i] + y[i + 1]);
        }
        sum
    }
}

/// Composite Simpson's rule integration along an array.
pub fn simpson(y: &Array1<f64>, x: Option<&Array1<f64>>, dx: Option<f64>) -> f64 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }
    if n == 2 {
        return trapz(y, x, dx);
    }

    if let Some(x_arr) = x {
        assert_eq!(x_arr.len(), n, "x and y must have the same length");
        let mut sum = 0.0;
        let mut i = 0;
        while i + 2 < n {
            let h0 = x_arr[i + 1] - x_arr[i];
            let h1 = x_arr[i + 2] - x_arr[i + 1];
            let y0 = y[i];
            let y1 = y[i + 1];
            let y2 = y[i + 2];

            // Simpson's rule for non-uniform spacing
            let term = (h0 + h1)
                / 6.0
                * ((2.0 - h1 / h0) * y0
                    + (h0 + h1).powi(2) / (h0 * h1) * y1
                    + (2.0 - h0 / h1) * y2);
            sum += term;
            i += 2;
        }

        // If odd number of intervals (even number of points), use trapezoid for last panel
        if i + 1 < n {
            let h = x_arr[i + 1] - x_arr[i];
            sum += 0.5 * h * (y[i] + y[i + 1]);
        }
        sum
    } else {
        let h = dx.unwrap_or(1.0);
        let mut sum = 0.0;
        let mut i = 0;
        while i + 2 < n {
            sum += (h / 3.0) * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
            i += 2;
        }
        if i + 1 < n {
            sum += 0.5 * h * (y[i] + y[i + 1]);
        }
        sum
    }
}

/// Cumulative trapezoidal integration.
pub fn cumtrapz(
    y: &Array1<f64>,
    x: Option<&Array1<f64>>,
    dx: Option<f64>,
    initial: Option<f64>,
) -> Array1<f64> {
    let n = y.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    let mut out = Vec::with_capacity(n);
    let mut acc = initial.unwrap_or(0.0);
    out.push(acc);

    if let Some(x_arr) = x {
        assert_eq!(x_arr.len(), n, "x and y must have the same length");
        for i in 0..n - 1 {
            let h = x_arr[i + 1] - x_arr[i];
            acc += 0.5 * h * (y[i] + y[i + 1]);
            out.push(acc);
        }
    } else {
        let step = dx.unwrap_or(1.0);
        for i in 0..n - 1 {
            acc += 0.5 * step * (y[i] + y[i + 1]);
            out.push(acc);
        }
    }

    Array1::from_vec(out)
}

/// Romberg integration of callable `f` over `[a, b]`.
pub fn romberg<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: Option<f64>,
    max_steps: Option<usize>,
) -> Result<f64, String> {
    let tol = tol.unwrap_or(1.49e-8);
    let max_k = max_steps.unwrap_or(10).min(20);

    let mut r = vec![vec![0.0; max_k + 1]; max_k + 1];
    let h = b - a;
    r[0][0] = 0.5 * h * (f(a) + f(b));

    for i in 1..=max_k {
        let step_size = h / (1 << i) as f64;
        let mut sum = 0.0;
        let num_points = 1 << (i - 1);
        for k in 1..=num_points {
            let x = a + (2 * k - 1) as f64 * step_size;
            sum += f(x);
        }
        r[i][0] = 0.5 * r[i - 1][0] + step_size * sum;

        for j in 1..=i {
            let factor = 4.0_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }

        if (r[i][i] - r[i - 1][i - 1]).abs() < tol {
            return Ok(r[i][i]);
        }
    }

    Ok(r[max_k][max_k])
}

/// Result of an ODE initial value problem solver.
#[derive(Debug, Clone)]
pub struct OdeResult {
    /// Time points evaluated
    pub t: Vec<f64>,
    /// State vectors at each time point
    pub y: Vec<Array1<f64>>,
    /// Whether the solver terminated successfully
    pub success: bool,
    /// Status message
    pub message: String,
}

/// ODE initial value problem solver (Runge-Kutta 45 / Dormand-Prince).
///
/// Solves $dy/dt = f(t, y)$ from $t_{span}.0$ to $t_{span}.1$ starting at $y_0$.
pub fn solve_ivp<F>(
    f: F,
    t_span: (f64, f64),
    y0: &Array1<f64>,
    rtol: Option<f64>,
    atol: Option<f64>,
    max_step: Option<f64>,
) -> OdeResult
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
{
    let (t0, tf) = t_span;
    let rtol = rtol.unwrap_or(1e-3);
    let atol = atol.unwrap_or(1e-6);
    let max_step = max_step.unwrap_or((tf - t0).abs() / 10.0);

    let dim = y0.len();
    let mut t = t0;
    let mut y = y0.clone();

    let mut t_vec = vec![t];
    let mut y_vec = vec![y.clone()];

    let mut h = ((tf - t0) / 100.0).clamp(1e-6, max_step);
    let max_iter = 100_000;
    let mut iter = 0;

    // Dormand-Prince coefficients (RK45)
    let c2 = 1.0 / 5.0;
    let a21 = 1.0 / 5.0;

    let c3 = 3.0 / 10.0;
    let a31 = 3.0 / 40.0;
    let a32 = 9.0 / 40.0;

    let c4 = 4.0 / 5.0;
    let a41 = 44.0 / 45.0;
    let a42 = -56.0 / 15.0;
    let a43 = 32.0 / 9.0;

    let c5 = 8.0 / 9.0;
    let a51 = 19372.0 / 6561.0;
    let a52 = -25360.0 / 2187.0;
    let a53 = 64448.0 / 6561.0;
    let a54 = -212.0 / 729.0;

    let a61 = 9017.0 / 3168.0;
    let a62 = -355.0 / 33.0;
    let a63 = 46732.0 / 5247.0;
    let a64 = 49.0 / 176.0;
    let a65 = -5103.0 / 18656.0;

    let b1 = 35.0 / 384.0;
    let b3 = 500.0 / 1113.0;
    let b4 = 125.0 / 192.0;
    let b5 = -2187.0 / 6784.0;
    let b6 = 11.0 / 84.0;

    let e1 = 71.0 / 57600.0;
    let e3 = -71.0 / 16695.0;
    let e4 = 71.0 / 1920.0;
    let e5 = -17253.0 / 339200.0;
    let e6 = 22.0 / 525.0;
    let e7 = -1.0 / 40.0;

    while (tf - t).abs() > 1e-12 && iter < max_iter {
        iter += 1;
        if (t + h - tf).abs() < 1e-12 || (tf > t0 && t + h > tf) || (tf < t0 && t + h < tf) {
            h = tf - t;
        }

        let k1 = f(t, &y);
        let k2 = f(t + c2 * h, &(&y + &(&k1 * (h * a21))));
        let k3 = f(t + c3 * h, &(&y + &(&k1 * (h * a31) + &k2 * (h * a32))));
        let k4 = f(
            t + c4 * h,
            &(&y + &(&k1 * (h * a41) + &k2 * (h * a42) + &k3 * (h * a43))),
        );
        let k5 = f(
            t + c5 * h,
            &(&y + &(&k1 * (h * a51) + &k2 * (h * a52) + &k3 * (h * a53) + &k4 * (h * a54))),
        );
        let k6 = f(
            t + h,
            &(&y + &(&k1 * (h * a61)
                + &k2 * (h * a62)
                + &k3 * (h * a63)
                + &k4 * (h * a64)
                + &k5 * (h * a65))),
        );

        let y_next = &y + &(&(&k1 * b1 + &k3 * b3 + &k4 * b4 + &k5 * b5 + &k6 * b6) * h);
        let k7 = f(t + h, &y_next);

        // Error estimate
        let err_vec = &(&k1 * e1 + &k3 * e3 + &k4 * e4 + &k5 * e5 + &k6 * e6 + &k7 * e7) * h;

        let mut max_err_ratio: f64 = 0.0;
        for i in 0..dim {
            let sc = atol + rtol * y[i].abs().max(y_next[i].abs());
            let ratio = err_vec[i].abs() / sc;
            if ratio > max_err_ratio {
                max_err_ratio = ratio;
            }
        }

        if max_err_ratio <= 1.0 {
            // Step accepted
            t += h;
            y = y_next;
            t_vec.push(t);
            y_vec.push(y.clone());

            let factor = if max_err_ratio > 0.0 {
                (0.9 / max_err_ratio.powf(0.2)).clamp(0.2, 5.0)
            } else {
                2.0
            };
            h = (h * factor).min(max_step);
        } else {
            // Step rejected, reduce step size
            let factor = (0.9 / max_err_ratio.powf(0.2)).clamp(0.1, 0.9);
            h *= factor;
        }

        if h.abs() < 1e-15 {
            break;
        }
    }

    let success = (tf - t).abs() <= 1e-6;
    OdeResult {
        t: t_vec,
        y: y_vec,
        success,
        message: if success {
            "Integration successful.".to_string()
        } else {
            "Required step size is less than spacing between numbers.".to_string()
        },
    }
}

/// SciPy-compatible `odeint` wrapper integrating `func(y, t)` at points `t`.
pub fn odeint<F>(func: F, y0: &Array1<f64>, t: &[f64]) -> Result<Array2<f64>, String>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64>,
{
    if t.is_empty() {
        return Err("t must have at least 1 element".to_string());
    }
    let n_steps = t.len();
    let dim = y0.len();
    let mut out = Array2::<f64>::zeros((n_steps, dim));

    for d in 0..dim {
        out[[0, d]] = y0[d];
    }

    let mut current_y = y0.clone();
    for i in 0..n_steps - 1 {
        let t_start = t[i];
        let t_end = t[i + 1];
        let f_adapted = |time: f64, y: &Array1<f64>| func(y, time);
        let res = solve_ivp(f_adapted, (t_start, t_end), &current_y, None, None, None);
        if !res.success {
            return Err(res.message);
        }
        current_y = res.y.last().unwrap().clone();
        for d in 0..dim {
            out[[i + 1, d]] = current_y[d];
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_quad_polynomial_and_trig() {
        // \int_0^1 x^2 dx = 1/3
        let (val, err) = quad(|x| x * x, 0.0, 1.0, None, None).unwrap();
        assert!((val - 1.0 / 3.0).abs() < 1e-7, "val = {}", val);
        assert!(err < 1e-5);

        // \int_0^pi sin(x) dx = 2
        let (val_sin, _) = quad(|x| x.sin(), 0.0, PI, None, None).unwrap();
        assert!((val_sin - 2.0).abs() < 1e-7, "val_sin = {}", val_sin);
    }

    #[test]
    fn test_dblquad() {
        // \int_0^1 \int_0^1 x * y dy dx = 1/4
        let (val, _) = dblquad(|y, x| x * y, 0.0, 1.0, |_| 0.0, |_| 1.0, None, None).unwrap();
        assert!((val - 0.25).abs() < 1e-5, "val = {}", val);
    }

    #[test]
    fn test_trapz_and_simpson() {
        // y = x^2 on [0, 1] with 5 points
        let x = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let y = Array1::from_vec(vec![0.0, 0.0625, 0.25, 0.5625, 1.0]);

        let trap_res = trapz(&y, Some(&x), None);
        // Trapezoid approximation: (0.0 + 2*(0.0625 + 0.25 + 0.5625) + 1.0) * 0.25 / 2 = 0.34375
        assert!((trap_res - 0.34375).abs() < 1e-6);

        let simp_res = simpson(&y, Some(&x), None);
        // Simpson is exact for polynomials up to degree 3!
        assert!((simp_res - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cumtrapz() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let cum = cumtrapz(&y, Some(&x), None, None);
        assert_eq!(cum.to_vec(), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_romberg() {
        // \int_0^1 e^x dx = e - 1
        let val = romberg(|x| x.exp(), 0.0, 1.0, None, None).unwrap();
        let exact = 1.0_f64.exp() - 1.0;
        assert!((val - exact).abs() < 1e-7);
    }

    #[test]
    fn test_solve_ivp_and_odeint() {
        // Exponential decay: dy/dt = -y, y(0) = 1 => y(t) = exp(-t)
        let y0 = Array1::from_vec(vec![1.0]);
        let res = solve_ivp(|_t, y| -y, (0.0, 2.0), &y0, None, None, None);
        assert!(res.success);
        let final_y = res.y.last().unwrap()[0];
        let exact = (-2.0_f64).exp();
        assert!((final_y - exact).abs() < 1e-3);

        // odeint
        let t_grid = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let sol = odeint(|y, _t| -y, &y0, &t_grid).unwrap();
        assert_eq!(sol.dim(), (5, 1));
        assert!((sol[[4, 0]] - exact).abs() < 1e-3);
    }
}
