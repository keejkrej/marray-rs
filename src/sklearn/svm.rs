use ndarray::{Array1, Array2};

fn kernel_linear(xi: &Array1<f64>, xj: &Array1<f64>) -> f64 {
    xi.dot(xj)
}

fn kernel_rbf(xi: &Array1<f64>, xj: &Array1<f64>, gamma: f64) -> f64 {
    let mut dist_sq = 0.0;
    for k in 0..xi.len() {
        let d = xi[k] - xj[k];
        dist_sq += d * d;
    }
    (-gamma * dist_sq).exp()
}

fn kernel_poly(xi: &Array1<f64>, xj: &Array1<f64>, gamma: f64, degree: usize) -> f64 {
    (gamma * xi.dot(xj) + 1.0).powi(degree as i32)
}

fn kernel_eval(xi: &Array1<f64>, xj: &Array1<f64>, kernel: &str, gamma: f64, degree: usize) -> f64 {
    match kernel {
        "linear" => kernel_linear(xi, xj),
        "rbf" => kernel_rbf(xi, xj, gamma),
        "poly" => kernel_poly(xi, xj, gamma, degree),
        _ => panic!("Unsupported kernel: {kernel}"),
    }
}

/// Support Vector Classifier using the SMO algorithm.
pub struct SVC {
    pub kernel: String,
    pub c: f64,
    pub gamma: f64,
    pub degree: usize,
    pub tol: f64,
    pub max_iter: usize,
    support_vectors: Option<Array2<f64>>,
    alphas: Option<Vec<f64>>,
    bias: f64,
    y_train: Option<Vec<f64>>,
}

impl SVC {
    pub fn new(kernel: &str, c: f64, gamma: f64, degree: usize) -> Self {
        Self {
            kernel: kernel.to_string(),
            c,
            gamma,
            degree,
            tol: 1e-3,
            max_iter: 1000,
            support_vectors: None,
            alphas: None,
            bias: 0.0,
            y_train: None,
        }
    }

    fn compute_error(alphas: &[f64], y: &[f64], kernel_matrix: &[Vec<f64>], bias: f64, i: usize) -> f64 {
        let mut sum = 0.0;
        for j in 0..alphas.len() {
            sum += alphas[j] * y[j] * kernel_matrix[i][j];
        }
        sum + bias - y[i]
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let n = x.nrows();
        assert_eq!(n, y.len(), "X and y must have the same number of samples");

        let y_vec: Vec<f64> = y.iter().copied().collect();
        let x_rows: Vec<Array1<f64>> = (0..n).map(|i| x.row(i).to_owned()).collect();

        let mut kernel_matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let k = kernel_eval(&x_rows[i], &x_rows[j], &self.kernel, self.gamma, self.degree);
                kernel_matrix[i][j] = k;
                kernel_matrix[j][i] = k;
            }
        }

        let mut alphas = vec![0.0f64; n];
        let mut bias = 0.0f64;

        for _iter in 0..self.max_iter {
            let mut num_changed = 0;

            for i in 0..n {
                let ei = Self::compute_error(&alphas, &y_vec, &kernel_matrix, bias, i);

                let violate_kkt = (y_vec[i] * ei < -self.tol && alphas[i] < self.c)
                    || (y_vec[i] * ei > self.tol && alphas[i] > 0.0);

                if !violate_kkt {
                    continue;
                }

                let mut best_j = 0;
                let mut max_diff = 0.0f64;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let ej = Self::compute_error(&alphas, &y_vec, &kernel_matrix, bias, j);
                    let diff = (ei - ej).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        best_j = j;
                    }
                }
                let j = best_j;
                let ej = Self::compute_error(&alphas, &y_vec, &kernel_matrix, bias, j);

                let alpha_i_old = alphas[i];
                let alpha_j_old = alphas[j];

                let (low, high) = if (y_vec[i] - y_vec[j]).abs() < 1e-12 {
                    let l = (0.0f64).max(alpha_i_old + alpha_j_old - self.c);
                    let h = self.c.min(alpha_i_old + alpha_j_old);
                    (l, h)
                } else {
                    let l = 0.0f64.max(alpha_j_old - alpha_i_old);
                    let h = self.c.min(self.c + alpha_j_old - alpha_i_old);
                    (l, h)
                };

                if (low - high).abs() < 1e-12 {
                    continue;
                }

                let eta = kernel_matrix[i][i] + kernel_matrix[j][j] - 2.0 * kernel_matrix[i][j];
                if eta <= 0.0 {
                    continue;
                }

                alphas[j] = alpha_j_old + y_vec[j] * (ei - ej) / eta;
                if alphas[j] > high {
                    alphas[j] = high;
                }
                if alphas[j] < low {
                    alphas[j] = low;
                }

                if (alphas[j] - alpha_j_old).abs() < 1e-5 {
                    continue;
                }

                alphas[i] = alpha_i_old + y_vec[i] * y_vec[j] * (alpha_j_old - alphas[j]);

                let b1 = bias - ei
                    - y_vec[i] * (alphas[i] - alpha_i_old) * kernel_matrix[i][i]
                    - y_vec[j] * (alphas[j] - alpha_j_old) * kernel_matrix[i][j];
                let b2 = bias - ej
                    - y_vec[i] * (alphas[i] - alpha_i_old) * kernel_matrix[i][j]
                    - y_vec[j] * (alphas[j] - alpha_j_old) * kernel_matrix[j][j];

                if alphas[i] > 0.0 && alphas[i] < self.c {
                    bias = b1;
                } else if alphas[j] > 0.0 && alphas[j] < self.c {
                    bias = b2;
                } else {
                    bias = (b1 + b2) / 2.0;
                }

                num_changed += 1;
            }

            if num_changed == 0 {
                break;
            }
        }

        let sv_indices: Vec<usize> = (0..n).filter(|&i| alphas[i] > 1e-10).collect();
        let sv_count = sv_indices.len();
        let ncols = x.ncols();

        let mut sv = Array2::zeros((sv_count, ncols));
        let mut sv_alphas = Vec::with_capacity(sv_count);
        let mut sv_y = Vec::with_capacity(sv_count);
        for (k, &i) in sv_indices.iter().enumerate() {
            for c in 0..ncols {
                sv[[k, c]] = x[[i, c]];
            }
            sv_alphas.push(alphas[i]);
            sv_y.push(y_vec[i]);
        }

        self.support_vectors = Some(sv);
        self.alphas = Some(sv_alphas);
        self.bias = bias;
        self.y_train = Some(sv_y);
    }

    pub fn decision_function(&self, x: &Array2<f64>) -> Array1<f64> {
        let sv = self.support_vectors.as_ref().expect("SVC not fitted");
        let alphas = self.alphas.as_ref().expect("SVC not fitted");
        let y_sv = self.y_train.as_ref().expect("SVC not fitted");
        let n_sv = sv.nrows();

        let sv_rows: Vec<Array1<f64>> = (0..n_sv).map(|i| sv.row(i).to_owned()).collect();

        let mut decisions = Array1::zeros(x.nrows());
        for r in 0..x.nrows() {
            let xr = x.row(r).to_owned();
            let mut sum = 0.0;
            for k in 0..n_sv {
                sum += alphas[k] * y_sv[k] * kernel_eval(&sv_rows[k], &xr, &self.kernel, self.gamma, self.degree);
            }
            decisions[r] = sum + self.bias;
        }
        decisions
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let decisions = self.decision_function(x);
        decisions.mapv(|v| if v >= 0.0 { 1.0 } else { -1.0 })
    }
}

/// Support Vector Regressor using epsilon-insensitive loss and the SMO algorithm.
pub struct SVR {
    pub kernel: String,
    pub c: f64,
    pub epsilon: f64,
    pub gamma: f64,
    pub degree: usize,
    pub tol: f64,
    pub max_iter: usize,
    support_vectors: Option<Array2<f64>>,
    alphas: Option<Vec<f64>>,
    bias: f64,
    sv_y: Option<Vec<f64>>,
}

impl SVR {
    pub fn new(kernel: &str, c: f64, epsilon: f64, gamma: f64) -> Self {
        Self {
            kernel: kernel.to_string(),
            c,
            epsilon,
            gamma,
            degree: 3,
            tol: 1e-3,
            max_iter: 1000,
            support_vectors: None,
            alphas: None,
            bias: 0.0,
            sv_y: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let n = x.nrows();
        assert_eq!(n, y.len(), "X and y must have the same number of samples");

        let y_vec: Vec<f64> = y.iter().copied().collect();
        let x_rows: Vec<Array1<f64>> = (0..n).map(|i| x.row(i).to_owned()).collect();

        let mut kernel_matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let k = kernel_eval(&x_rows[i], &x_rows[j], &self.kernel, self.gamma, self.degree);
                kernel_matrix[i][j] = k;
                kernel_matrix[j][i] = k;
            }
        }

        let mut alpha = vec![0.0f64; n];
        let mut alpha_star = vec![0.0f64; n];
        let mut bias = 0.0f64;

        for _iter in 0..self.max_iter {
            let mut num_changed = 0;

            for i in 0..n {
                let mut pred_i = bias;
                for j in 0..n {
                    pred_i += (alpha[j] - alpha_star[j]) * kernel_matrix[i][j];
                }
                let ei = pred_i - y_vec[i];

                let violate_pos = ei > self.epsilon && alpha[i] < self.c;
                let violate_neg = ei < -self.epsilon && alpha_star[i] < self.c;

                if !violate_pos && !violate_neg {
                    continue;
                }

                let mut best_j = 0;
                let mut max_diff = 0.0f64;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let mut pred_j = bias;
                    for k in 0..n {
                        pred_j += (alpha[k] - alpha_star[k]) * kernel_matrix[j][k];
                    }
                    let ej = pred_j - y_vec[j];
                    let diff = (ei - ej).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        best_j = j;
                    }
                }
                let j = best_j;

                let mut pred_j = bias;
                for k in 0..n {
                    pred_j += (alpha[k] - alpha_star[k]) * kernel_matrix[j][k];
                }
                let ej = pred_j - y_vec[j];

                let eta = kernel_matrix[i][i] + kernel_matrix[j][j] - 2.0 * kernel_matrix[i][j];
                if eta <= 0.0 {
                    continue;
                }

                let alpha_i_old = alpha[i];
                let alpha_star_i_old = alpha_star[i];
                let alpha_j_old = alpha[j];
                let alpha_star_j_old = alpha_star[j];

                if violate_pos {
                    let delta = (ei - self.epsilon) / eta;
                    let new_alpha_j = (alpha_j_old + delta).clamp(0.0, self.c);
                    let change = new_alpha_j - alpha_j_old;
                    alpha[j] = new_alpha_j;
                    alpha[i] = (alpha_i_old - change).clamp(0.0, self.c);
                } else {
                    let delta = (-ei - self.epsilon) / eta;
                    let new_alpha_star_j = (alpha_star_j_old + delta).clamp(0.0, self.c);
                    let change = new_alpha_star_j - alpha_star_j_old;
                    alpha_star[j] = new_alpha_star_j;
                    alpha_star[i] = (alpha_star_i_old - change).clamp(0.0, self.c);
                }

                if (alpha[j] - alpha_j_old).abs() < 1e-5
                    && (alpha_star[j] - alpha_star_j_old).abs() < 1e-5
                    && (alpha[i] - alpha_i_old).abs() < 1e-5
                    && (alpha_star[i] - alpha_star_i_old).abs() < 1e-5
                {
                    continue;
                }

                let b1 = bias - ei + self.epsilon;
                let b2 = bias - ej + self.epsilon;
                let b3 = bias - ei - self.epsilon;
                let b4 = bias - ej - self.epsilon;

                if alpha[i] > 0.0 && alpha[i] < self.c {
                    bias = b1;
                } else if alpha_star[i] > 0.0 && alpha_star[i] < self.c {
                    bias = b3;
                } else if alpha[j] > 0.0 && alpha[j] < self.c {
                    bias = b2;
                } else if alpha_star[j] > 0.0 && alpha_star_j_old < self.c {
                    bias = b4;
                } else {
                    bias = (b1 + b2 + b3 + b4) / 4.0;
                }

                num_changed += 1;
            }

            if num_changed == 0 {
                break;
            }
        }

        let net_alpha: Vec<f64> = alpha.iter().zip(alpha_star.iter()).map(|(a, b)| a - b).collect();
        let sv_indices: Vec<usize> = (0..n).filter(|&i| net_alpha[i].abs() > 1e-10).collect();
        let sv_count = sv_indices.len();
        let ncols = x.ncols();

        let mut sv = Array2::zeros((sv_count, ncols));
        let mut sv_alphas = Vec::with_capacity(sv_count);
        let mut sv_y_vals = Vec::with_capacity(sv_count);
        for (k, &i) in sv_indices.iter().enumerate() {
            for c in 0..ncols {
                sv[[k, c]] = x[[i, c]];
            }
            sv_alphas.push(net_alpha[i]);
            sv_y_vals.push(y_vec[i]);
        }

        self.support_vectors = Some(sv);
        self.alphas = Some(sv_alphas);
        self.bias = bias;
        self.sv_y = Some(sv_y_vals);
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let sv = self.support_vectors.as_ref().expect("SVR not fitted");
        let alphas = self.alphas.as_ref().expect("SVR not fitted");
        let n_sv = sv.nrows();

        let sv_rows: Vec<Array1<f64>> = (0..n_sv).map(|i| sv.row(i).to_owned()).collect();

        let mut preds = Array1::zeros(x.nrows());
        for r in 0..x.nrows() {
            let xr = x.row(r).to_owned();
            let mut sum = 0.0;
            for k in 0..n_sv {
                sum += alphas[k] * kernel_eval(&sv_rows[k], &xr, &self.kernel, self.gamma, self.degree);
            }
            preds[r] = sum + self.bias;
        }
        preds
    }
}

/// Linear SVM using gradient descent on the primal hinge loss.
pub struct LinearSVC {
    pub c: f64,
    pub max_iter: usize,
    pub tol: f64,
    weights: Option<Array1<f64>>,
    bias: f64,
}

impl LinearSVC {
    pub fn new(c: f64, max_iter: usize) -> Self {
        Self {
            c,
            max_iter,
            tol: 1e-4,
            weights: None,
            bias: 0.0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (nrows, ncols) = x.dim();
        assert_eq!(nrows, y.len(), "X and y must have the same number of samples");

        let y_binary: Array1<f64> = y.mapv(|v| if v <= 0.0 { -1.0 } else { 1.0 });

        let mut w = Array1::zeros(ncols);
        let mut b = 0.0f64;
        let lr = 1.0;

        for epoch in 0..self.max_iter {
            let mut grad_w = Array1::zeros(ncols);
            let mut grad_b = 0.0f64;

            for i in 0..nrows {
                let xi = x.row(i);
                let yi = y_binary[i];
                let margin = yi * (xi.dot(&w) + b);

                if margin < 1.0 {
                    for c in 0..ncols {
                        grad_w[c] += -yi * xi[c];
                    }
                    grad_b += -yi;
                }
            }

            let scale = self.c / nrows as f64;
            for c in 0..ncols {
                grad_w[c] = w[c] + scale * grad_w[c];
            }
            grad_b *= scale;

            let mut step_size = lr / (1.0 + epoch as f64 * 0.01);
            let grad_norm: f64 = grad_w.iter().map(|v: &f64| v * v).sum::<f64>().sqrt() + grad_b.abs();
            if grad_norm > 1.0 {
                step_size /= grad_norm;
            }

            for c in 0..ncols {
                w[c] -= step_size * grad_w[c];
            }
            b -= step_size * grad_b;

            if epoch > 0 && epoch % 100 == 0 {
                let mut total_grad = 0.0f64;
                for c in 0..ncols {
                    total_grad += grad_w[c].abs();
                }
                total_grad += grad_b.abs();
                if total_grad < self.tol {
                    break;
                }
            }
        }

        self.weights = Some(w);
        self.bias = b;
    }

    pub fn decision_function(&self, x: &Array2<f64>) -> Array1<f64> {
        let w = self.weights.as_ref().expect("LinearSVC not fitted");
        let mut scores = x.dot(w);
        scores.mapv_inplace(|v| v + self.bias);
        scores
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let scores = self.decision_function(x);
        scores.mapv(|v| if v >= 0.0 { 1.0 } else { -1.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn make_linear_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0,
                2.0, 2.0,
                1.5, 1.5,
                2.0, 1.0,
                -1.0, -1.0,
                -2.0, -2.0,
                -1.5, -1.5,
                -2.0, -1.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        (x, y)
    }

    fn make_circles_data() -> (Array2<f64>, Array1<f64>) {
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for i in 0..40 {
            let angle = std::f64::consts::PI * 2.0 * i as f64 / 40.0;
            let (r, label) = if i % 2 == 0 { (0.5, -1.0) } else { (1.5, 1.0) };
            x_data.push(r * angle.cos());
            x_data.push(r * angle.sin());
            y_data.push(label);
        }
        let x = Array2::from_shape_vec((40, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        (x, y)
    }

    #[test]
    fn test_svc_linear() {
        let (x, y) = make_linear_data();
        let mut clf = SVC::new("linear", 1.0, 1.0, 3);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        for i in 0..y.len() {
            assert_eq!(preds[i], y[i], "Misclassified sample {i}");
        }
    }

    #[test]
    fn test_svc_rbf() {
        let (x, y) = make_circles_data();
        let mut clf = SVC::new("rbf", 10.0, 2.0, 3);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, y)| **p == **y).count();
        assert!(
            correct as f64 / y.len() as f64 > 0.85,
            "RBF SVC accuracy too low: {}/{}",
            correct,
            y.len()
        );
    }

    #[test]
    fn test_svc_poly() {
        let (x, y) = make_linear_data();
        let mut clf = SVC::new("poly", 1.0, 1.0, 2);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        for i in 0..y.len() {
            assert_eq!(preds[i], y[i], "Misclassified sample {i} with poly kernel");
        }
    }

    #[test]
    fn test_svr() {
        let n = 50;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        for i in 0..n {
            let xi = i as f64 * 0.1;
            let yi = xi.sin() + (i as f64 * 0.37).sin() * 0.1;
            x_data.push(xi);
            y_data.push(yi);
        }
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let mut reg = SVR::new("rbf", 10.0, 0.05, 1.0);
        reg.fit(&x, &y);

        let preds = reg.predict(&x);
        let mse: f64 = preds.iter().zip(y.iter()).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / n as f64;
        assert!(mse < 0.1, "SVR MSE too high: {mse}");
    }

    #[test]
    fn test_linear_svc() {
        let (x, y) = make_linear_data();
        let mut clf = LinearSVC::new(1.0, 2000);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        for i in 0..y.len() {
            assert_eq!(preds[i], y[i], "LinearSVC misclassified sample {i}");
        }
    }

    #[test]
    fn test_linear_svc_nontrivial() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.5, 0.5,
                1.0, 0.2,
                0.3, 0.8,
                0.8, 0.9,
                0.1, 0.4,
                -0.5, -0.5,
                -1.0, -0.2,
                -0.3, -0.8,
                -0.8, -0.9,
                -0.1, -0.4,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        let mut clf = LinearSVC::new(1.0, 5000);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        for i in 0..y.len() {
            assert_eq!(preds[i], y[i], "LinearSVC misclassified sample {i}");
        }
    }
}
