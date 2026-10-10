use ndarray::{Array1, Array2};

/// Golden-section search for a 1D scalar minimization problem.
pub fn rsminimize_scalar<F>(f: F, bounds: (f64, f64), tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let (mut a, mut b) = bounds;
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / golden_ratio;

    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    while (b - a).abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = f(d);
        }
    }

    (a + b) / 2.0
}

/// Find a root of a scalar function in a bracketing interval using Brent's method.
pub fn brentq<F>(f: F, mut a: f64, mut b: f64, xtol: f64, rtol: f64, max_iter: usize) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);

    if fa * fb > 0.0 {
        return Err(format!("Root is not bracketed: f(a)={}, f(b)={}", fa, fb));
    }
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    for _ in 0..max_iter {
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * xtol + rtol * b.abs();
        let m = 0.5 * (c - b);

        if m.abs() <= tol || fb == 0.0 {
            return Ok(b);
        }

        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q);
            if (a - c).abs() < f64::EPSILON {
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }

        a = b;
        fa = fb;
        if d.abs() > tol {
            b += d;
        } else if m > 0.0 {
            b += tol;
        } else {
            b -= tol;
        }
        fb = f(b);
    }

    Ok(b)
}

/// Find a root of a scalar function using Newton-Raphson.
pub fn newton<F, Fp>(f: F, fprime: Option<Fp>, mut x0: f64, tol: f64, max_iter: usize) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
    Fp: Fn(f64) -> f64,
{
    let eps = 1e-7;
    for _ in 0..max_iter {
        let fx = f(x0);
        if fx.abs() < tol {
            return Ok(x0);
        }
        let df = match &fprime {
            Some(fp) => fp(x0),
            None => (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps),
        };
        if df.abs() < 1e-15 {
            return Err("Derivative too small in Newton-Raphson".to_string());
        }
        let x1 = x0 - fx / df;
        if (x1 - x0).abs() < tol {
            return Ok(x1);
        }
        x0 = x1;
    }
    Ok(x0)
}

/// Fit model parameters using a simple Gauss-Newton least-squares update.
pub fn curve_fit<F>(f: F, xdata: &Array1<f64>, ydata: &Array1<f64>, p0: &Array1<f64>) -> Result<Array1<f64>, String>
where
    F: Fn(&Array1<f64>, f64) -> f64,
{
    let n = xdata.len();
    let p = p0.len();
    if n == 0 || p == 0 {
        return Ok(p0.clone());
    }
    assert_eq!(n, ydata.len(), "xdata and ydata must have the same length");

    let mut params = p0.to_vec();
    let eps = 1e-6;
    for _ in 0..50 {
        let mut jtj = Array2::<f64>::zeros((p, p));
        let mut jtr = Array1::<f64>::zeros(p);
        for i in 0..n {
            let yhat = f(&Array1::from_vec(params.clone()), xdata[i]);
            let residual = ydata[i] - yhat;
            for j in 0..p {
                let mut plus = params.clone();
                plus[j] += eps;
                let mut minus = params.clone();
                minus[j] -= eps;
                let jac = (f(&Array1::from_vec(plus), xdata[i]) - f(&Array1::from_vec(minus), xdata[i])) / (2.0 * eps);
                jtj[[j, j]] += jac * jac;
                jtr[j] += jac * residual;
            }
        }
        let mut delta = Array1::<f64>::zeros(p);
        for i in 0..p {
            delta[i] = jtr[i] / (jtj[[i, i]] + 1e-12);
        }
        let step_norm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
        for i in 0..p {
            params[i] += delta[i];
        }
        if step_norm < 1e-8 {
            break;
        }
    }
    Ok(Array1::from_vec(params))
}

/// Optimization result struct matching scipy.optimize.OptimizeResult.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub x: Array1<f64>,
    pub fun: f64,
    pub nit: usize,
    pub nfev: usize,
    pub success: bool,
    pub message: String,
}

/// Minimize a scalar function of one or more variables.
///
/// Methods supported:
/// - `"nelder-mead"`: Downhill simplex algorithm (derivative-free).
/// - `"bfgs"`: Broyden-Fletcher-Goldfarb-Shanno quasi-Newton algorithm.
pub fn minimize<F>(
    f: F,
    x0: &Array1<f64>,
    method: &str,
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> Result<OptimizeResult, String>
where
    F: Fn(&Array1<f64>) -> f64,
{
    match method.to_lowercase().as_str() {
        "nelder-mead" => minimize_nelder_mead(f, x0, tol.unwrap_or(1e-5), max_iter.unwrap_or(1000)),
        "bfgs" => minimize_bfgs(f, x0, tol.unwrap_or(1e-5), max_iter.unwrap_or(200)),
        other => Err(format!("Unsupported optimization method: {}", other)),
    }
}

fn minimize_nelder_mead<F>(f: F, x0: &Array1<f64>, tol: f64, max_iter: usize) -> Result<OptimizeResult, String>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err("x0 must not be empty".to_string());
    }

    // Initialize simplex with n + 1 points
    let mut simplex: Vec<Array1<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());

    for i in 0..n {
        let mut pt = x0.clone();
        if pt[i].abs() > 1e-8 {
            pt[i] *= 1.05;
        } else {
            pt[i] = 0.00025;
        }
        simplex.push(pt);
    }

    let mut nfev = 0;
    let mut values: Vec<f64> = simplex
        .iter()
        .map(|pt| {
            nfev += 1;
            f(pt)
        })
        .collect();

    let alpha = 1.0;  // Reflection
    let gamma = 2.0;  // Expansion
    let rho = 0.5;    // Contraction
    let sigma = 0.5;  // Shrink

    let mut nit = 0;
    for iter in 0..max_iter {
        nit = iter + 1;

        // Sort simplex by function values
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));

        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        values = order.iter().map(|&i| values[i]).collect();

        // Check convergence (standard deviation of simplex values)
        let mean_val = values.iter().sum::<f64>() / ((n + 1) as f64);
        let var = values.iter().map(|v| (v - mean_val).powi(2)).sum::<f64>() / ((n + 1) as f64);
        if var.sqrt() < tol {
            return Ok(OptimizeResult {
                x: simplex[0].clone(),
                fun: values[0],
                nit,
                nfev,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }

        // Compute centroid of first n points
        let mut centroid = Array1::zeros(n);
        for i in 0..n {
            centroid = &centroid + &simplex[i];
        }
        centroid.mapv_inplace(|v| v / n as f64);

        // 1. Reflection
        let xr = &centroid + &((&centroid - &simplex[n]) * alpha);
        nfev += 1;
        let fr = f(&xr);

        if fr < values[n - 1] && fr >= values[0] {
            simplex[n] = xr;
            values[n] = fr;
            continue;
        }

        // 2. Expansion
        if fr < values[0] {
            let xe = &centroid + &((&xr - &centroid) * gamma);
            nfev += 1;
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                values[n] = fe;
            } else {
                simplex[n] = xr;
                values[n] = fr;
            }
            continue;
        }

        // 3. Contraction
        if fr < values[n] {
            // Outside contraction
            let xc = &centroid + &((&xr - &centroid) * rho);
            nfev += 1;
            let fc = f(&xc);
            if fc <= fr {
                simplex[n] = xc;
                values[n] = fc;
                continue;
            }
        } else {
            // Inside contraction
            let xc = &centroid - &((&centroid - &simplex[n]) * rho);
            nfev += 1;
            let fc = f(&xc);
            if fc < values[n] {
                simplex[n] = xc;
                values[n] = fc;
                continue;
            }
        }

        // 4. Shrink
        let x0_best = simplex[0].clone();
        for i in 1..=n {
            simplex[i] = &x0_best + &((&simplex[i] - &x0_best) * sigma);
            nfev += 1;
            values[i] = f(&simplex[i]);
        }
    }

    Ok(OptimizeResult {
        x: simplex[0].clone(),
        fun: values[0],
        nit,
        nfev,
        success: false,
        message: "Maximum iterations reached.".to_string(),
    })
}

fn minimize_bfgs<F>(f: F, x0: &Array1<f64>, tol: f64, max_iter: usize) -> Result<OptimizeResult, String>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err("x0 must not be empty".to_string());
    }

    let mut x = x0.clone();
    let mut nfev = 0;
    let mut fx = f(&x);
    nfev += 1;

    let grad = |pt: &Array1<f64>, f_val: f64| -> (Array1<f64>, usize) {
        let eps = 1e-7;
        let mut g = Array1::zeros(n);
        let mut calls = 0;
        for i in 0..n {
            let mut p_plus = pt.clone();
            p_plus[i] += eps;
            let f_plus = f(&p_plus);
            calls += 1;
            g[i] = (f_plus - f_val) / eps;
        }
        (g, calls)
    };

    let (mut g, calls) = grad(&x, fx);
    nfev += calls;

    // Inverse Hessian approximation initialized to identity
    let mut h_inv = Array2::<f64>::eye(n);

    let mut nit = 0;
    for iter in 0..max_iter {
        nit = iter + 1;

        let g_norm = g.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
        if g_norm < tol {
            return Ok(OptimizeResult {
                x,
                fun: fx,
                nit,
                nfev,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
            });
        }

        // Search direction: p = -H_inv * g
        let p = -h_inv.dot(&g);

        // Backtracking line search with Armijo condition
        let mut alpha = 1.0;
        let c1 = 1e-4;
        let p_dot_g = p.dot(&g);
        let mut x_new = &x + &(&p * alpha);
        let mut f_new = f(&x_new);
        nfev += 1;

        while f_new > fx + c1 * alpha * p_dot_g && alpha > 1e-12 {
            alpha *= 0.5;
            x_new = &x + &(&p * alpha);
            f_new = f(&x_new);
            nfev += 1;
        }

        let s = &x_new - &x;
        let (g_new, g_calls) = grad(&x_new, f_new);
        nfev += g_calls;
        let y = &g_new - &g;

        let ys = y.dot(&s);
        if ys.abs() > 1e-12 {
            let rho = 1.0 / ys;
            let eye = Array2::<f64>::eye(n);

            let s_mat = Array2::from_shape_fn((n, 1), |(i, _)| s[i]);
            let y_mat = Array2::from_shape_fn((1, n), |(_, j)| y[j]);
            let sy_outer = s_mat.dot(&y_mat);

            let y_col = Array2::from_shape_fn((n, 1), |(i, _)| y[i]);
            let s_row = Array2::from_shape_fn((1, n), |(_, j)| s[j]);
            let ys_outer = y_col.dot(&s_row);

            let v1 = &eye - &(&sy_outer * rho);
            let v2 = &eye - &(&ys_outer * rho);
            let ss_outer = s_mat.dot(&s_row);

            h_inv = v1.dot(&h_inv).dot(&v2) + &ss_outer * rho;
        }

        x = x_new;
        fx = f_new;
        g = g_new;
    }

    Ok(OptimizeResult {
        x,
        fun: fx,
        nit,
        nfev,
        success: false,
        message: "Maximum iterations reached.".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_rsminimize_scalar() {
        let x = rsminimize_scalar(|x| (x - 2.0).powi(2), (0.0, 4.0), 1e-6);
        assert!((x - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_brentq() {
        let root = brentq(|x| x * x - 2.0, 0.0, 2.0, 1e-8, 1e-8, 100).unwrap();
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_newton() {
        let root = newton(|x| x.sin(), None::<fn(f64) -> f64>, 3.0, 1e-8, 50).unwrap();
        assert!((root - std::f64::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_minimize_nelder_mead() {
        // Rosenbrock function: (1 - x)^2 + 100(y - x^2)^2, min at (1, 1) = 0
        let res = minimize(
            |v| (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0].powi(2)).powi(2),
            &Array1::from_vec(vec![0.0, 0.0]),
            "nelder-mead",
            Some(1e-4),
            Some(2000),
        ).unwrap();

        assert!(res.success);
        assert!((res.x[0] - 1.0).abs() < 0.05);
        assert!((res.x[1] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_minimize_bfgs() {
        let res = minimize(
            |v| (v[0] - 3.0).powi(2) + (v[1] + 2.0).powi(2),
            &Array1::from_vec(vec![0.0, 0.0]),
            "bfgs",
            Some(1e-5),
            Some(100),
        ).unwrap();

        assert!(res.success);
        assert!((res.x[0] - 3.0).abs() < 1e-3);
        assert!((res.x[1] + 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_curve_fit() {
        let xdata = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let ydata = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let p0 = Array1::from_vec(vec![1.0, 1.0]);
        let fitted = curve_fit(|params, x| params[0] * x + params[1], &xdata, &ydata, &p0).unwrap();
        assert!((fitted[0] - 2.0).abs() < 0.2);
        assert!((fitted[1] - 1.0).abs() < 0.2);
    }
}
