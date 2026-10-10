use ndarray::{Array1, Array2};

/// Common trait for kernel functions.
pub trait Kernel {
    /// Compute the Gram matrix between rows of `x` and `y`.
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64>;

    /// String identifier for the kernel.
    fn id(&self) -> &str;
}

/// Linear (dot-product) kernel.
pub struct LinearKernel {
    pub c0: f64,
}

impl LinearKernel {
    pub fn new(c0: f64) -> Self {
        Self { c0 }
    }
}

impl Default for LinearKernel {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Kernel for LinearKernel {
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64> {
        let y = y.unwrap_or(x);
        x.dot(&y.t()) + self.c0
    }

    fn id(&self) -> &str {
        "LinearKernel"
    }
}

/// Polynomial kernel.
pub struct PolynomialKernel {
    pub d: usize,
    pub gamma: Option<f64>,
    pub c0: f64,
}

impl PolynomialKernel {
    pub fn new(d: usize, gamma: Option<f64>, c0: f64) -> Self {
        Self { d, gamma, c0 }
    }
}

impl Default for PolynomialKernel {
    fn default() -> Self {
        Self::new(3, None, 1.0)
    }
}

impl Kernel for PolynomialKernel {
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64> {
        let y = y.unwrap_or(x);
        let gamma = self.gamma.unwrap_or(1.0 / x.ncols() as f64);
        let base = gamma * x.dot(&y.t()) + self.c0;
        base.mapv(|v| v.powi(self.d as i32))
    }

    fn id(&self) -> &str {
        "PolynomialKernel"
    }
}

/// Radial basis function (RBF) / squared exponential kernel.
pub struct RBFKernel {
    pub sigma: Option<f64>,
}

impl RBFKernel {
    pub fn new(sigma: Option<f64>) -> Self {
        Self { sigma }
    }
}

impl Default for RBFKernel {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Kernel for RBFKernel {
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64> {
        let y = y.unwrap_or(x);
        let sigma = self.sigma.unwrap_or((x.ncols() as f64 / 2.0).sqrt());
        let dists = pairwise_l2_distances(&(x / sigma), &(y / sigma));
        dists.mapv(|v| (-0.5 * v * v).exp())
    }

    fn id(&self) -> &str {
        "RBFKernel"
    }
}

/// Matérn kernel.
pub struct MaternKernel {
    pub nu: f64,
    pub sigma: Option<f64>,
}

impl MaternKernel {
    pub fn new(nu: f64, sigma: Option<f64>) -> Self {
        Self { nu, sigma }
    }
}

impl Default for MaternKernel {
    fn default() -> Self {
        Self::new(1.5, None)
    }
}

impl Kernel for MaternKernel {
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64> {
        let y = y.unwrap_or(x);
        let sigma = self.sigma.unwrap_or((x.ncols() as f64 / 2.0).sqrt());
        let d = pairwise_l2_distances(&(x / sigma), &(y / sigma));

        if (self.nu - 0.5).abs() < 1e-6 {
            d.mapv(|v| (-v).exp())
        } else if (self.nu - 1.5).abs() < 1e-6 {
            let sqrt3 = 3.0_f64.sqrt();
            d.mapv(|v| {
                let scaled = sqrt3 * v;
                (1.0 + scaled) * (-scaled).exp()
            })
        } else if (self.nu - 2.5).abs() < 1e-6 {
            let sqrt5 = 5.0_f64.sqrt();
            d.mapv(|v| {
                let scaled = sqrt5 * v;
                (1.0 + scaled + 5.0 / 3.0 * v * v) * (-scaled).exp()
            })
        } else if self.nu.is_infinite() || self.nu > 100.0 {
            d.mapv(|v| (-0.5 * v * v).exp())
        } else {
            // General approximation fallback
            let sqrt3 = (2.0 * self.nu).sqrt();
            d.mapv(|v| {
                let scaled = sqrt3 * v;
                (1.0 + scaled) * (-scaled).exp()
            })
        }
    }

    fn id(&self) -> &str {
        "MaternKernel"
    }
}

/// Rational Quadratic kernel.
pub struct RationalQuadraticKernel {
    pub alpha: f64,
    pub sigma: Option<f64>,
}

impl RationalQuadraticKernel {
    pub fn new(alpha: f64, sigma: Option<f64>) -> Self {
        Self { alpha, sigma }
    }
}

impl Default for RationalQuadraticKernel {
    fn default() -> Self {
        Self::new(1.0, None)
    }
}

impl Kernel for RationalQuadraticKernel {
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64> {
        let y = y.unwrap_or(x);
        let sigma = self.sigma.unwrap_or((x.ncols() as f64 / 2.0).sqrt());
        let d = pairwise_l2_distances(&(x / sigma), &(y / sigma));
        let alpha = self.alpha;
        d.mapv(|v| (1.0 + (v * v) / (2.0 * alpha)).powf(-alpha))
    }

    fn id(&self) -> &str {
        "RationalQuadraticKernel"
    }
}

/// Initialize a kernel from a string identifier or use the default linear kernel.
pub fn kernel_initializer(param: Option<&str>) -> Result<Box<dyn Kernel>, String> {
    match param {
        None | Some("linear") | Some("LinearKernel") => Ok(Box::new(LinearKernel::default())),
        Some("polynomial") | Some("PolynomialKernel") => Ok(Box::new(PolynomialKernel::default())),
        Some("rbf") | Some("RBFKernel") => Ok(Box::new(RBFKernel::default())),
        Some("matern") | Some("MaternKernel") => Ok(Box::new(MaternKernel::default())),
        Some("rational_quadratic") | Some("RationalQuadraticKernel") => {
            Ok(Box::new(RationalQuadraticKernel::default()))
        }
        Some(other) => Err(format!("Unknown kernel: {}", other)),
    }
}

/// Pairwise Euclidean distances between rows of `x` and `y`.
pub fn pairwise_l2_distances(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let (n, _) = x.dim();
    let (m, _) = y.dim();

    let x_sq: Array1<f64> = x.rows().into_iter().map(|row| row.iter().map(|&v| v * v).sum()).collect();
    let y_sq: Array1<f64> = y.rows().into_iter().map(|row| row.iter().map(|&v| v * v).sum()).collect();

    let mut dists = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let dot: f64 = (0..x.ncols()).map(|k| x[[i, k]] * y[[j, k]]).sum();
            let d = (x_sq[i] - 2.0 * dot + y_sq[j]).max(0.0);
            dists[[i, j]] = d.sqrt();
        }
    }
    dists
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_matern_kernel() {
        let k_15 = MaternKernel::new(1.5, Some(1.0));
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let gram = k_15.kernel(&x, None);
        assert_eq!(gram.dim(), (2, 2));
        assert!((gram[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((gram[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(gram[[0, 1]] > 0.0 && gram[[0, 1]] < 1.0);

        let k_05 = MaternKernel::new(0.5, Some(1.0));
        let gram_05 = k_05.kernel(&x, None);
        assert!((gram_05[[0, 1]] - (-1.0_f64).exp()).abs() < 1e-6);

        let k_25 = MaternKernel::new(2.5, Some(1.0));
        let gram_25 = k_25.kernel(&x, None);
        assert!(gram_25[[0, 1]] > 0.0 && gram_25[[0, 1]] < 1.0);
    }

    #[test]
    fn test_rational_quadratic_kernel() {
        let rq = RationalQuadraticKernel::new(1.0, Some(1.0));
        let x = array![[0.0], [1.0]];
        let gram = rq.kernel(&x, None);
        assert_eq!(gram.dim(), (2, 2));
        assert!((gram[[0, 0]] - 1.0).abs() < 1e-10);
        // d = 1.0, alpha = 1.0 -> (1 + 1 / 2)^(-1) = (1.5)^(-1) = 2/3
        assert!((gram[[0, 1]] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_initializer() {
        let k1 = kernel_initializer(Some("matern")).unwrap();
        assert_eq!(k1.id(), "MaternKernel");

        let k2 = kernel_initializer(Some("rational_quadratic")).unwrap();
        assert_eq!(k2.id(), "RationalQuadraticKernel");
    }
}

