use ndarray::Array2;

use super::activations::Activation;

/// Trait for all loss functions.
pub trait Loss {
    /// Compute the loss between ground truth and predictions.
    fn loss(&self, y: &Array2<f64>, y_pred: &Array2<f64>) -> f64;
    /// Compute the gradient of the loss with respect to predictions.
    fn grad(&self, y: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64>;
    /// Return the name of the loss function.
    fn name(&self) -> &str;
}

/// Squared error (L2) loss: 0.5 * ||y_pred - y||²
pub struct SquaredError;

impl SquaredError {
    pub fn new() -> Self {
        SquaredError
    }
}

impl Loss for SquaredError {
    fn loss(&self, y: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let diff = y_pred - y;
        let n = y.len() as f64;
        0.5 * diff.iter().map(|v| v * v).sum::<f64>() / n
    }

    fn grad(&self, y: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64> {
        let n = y.len() as f64;
        (y_pred - y) / n
    }

    fn name(&self) -> &str {
        "SquaredError"
    }
}

/// Cross-entropy loss: -sum(y * log(y_pred))
pub struct CrossEntropy;

impl CrossEntropy {
    pub fn new() -> Self {
        CrossEntropy
    }
}

impl Loss for CrossEntropy {
    fn loss(&self, y: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let eps = f64::EPSILON;
        let clipped = y_pred.mapv(|v| v.max(eps).min(1.0 - eps));
        let log_pred = clipped.mapv(|v| v.ln());
        -(y * &log_pred).sum()
    }

    fn grad(&self, y: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64> {
        y_pred - y
    }

    fn name(&self) -> &str {
        "CrossEntropy"
    }
}

/// Binary cross-entropy loss with optional activation function for the output layer.
pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> Self {
        BinaryCrossEntropy
    }
}

impl BinaryCrossEntropy {
    /// Compute binary cross-entropy loss between y and y_pred.
    pub fn loss_with_activation(
        &self,
        y: &Array2<f64>,
        y_pred: &Array2<f64>,
        act_fn: &dyn Activation,
    ) -> f64 {
        let eps = f64::EPSILON;
        let a = act_fn.fn_(y_pred);
        let a_clipped = a.mapv(|v| v.max(eps).min(1.0 - eps));
        let loss = -y * &a_clipped.mapv(|v| v.ln())
            - &(1.0 - y) * &(1.0 - &a_clipped).mapv(|v| v.ln());
        loss.sum()
    }
}

/// Variational lower bound loss for a Bernoulli VAE.
pub struct VAELoss;

impl VAELoss {
    pub fn new() -> Self {
        VAELoss
    }

    /// Compute the VAE loss (reconstruction + KL divergence).
    pub fn loss(
        &self,
        y: &Array2<f64>,
        y_pred: &Array2<f64>,
        t_mean: &Array2<f64>,
        t_log_var: &Array2<f64>,
    ) -> f64 {
        let eps = f64::EPSILON;
        let y_pred_clipped = y_pred.mapv(|v| v.max(eps).min(1.0 - eps));
        // Reconstruction loss (binary cross-entropy)
        let rec_loss = -y * &y_pred_clipped.mapv(|v| v.ln())
            - &(1.0 - y) * &(1.0 - &y_pred_clipped).mapv(|v| v.ln());
        let rec_loss: f64 = rec_loss.sum_axis(ndarray::Axis(1)).sum();

        // KL divergence
        let kl_loss =
            -0.5 * (1.0 + t_log_var - t_mean.mapv(|v| v * v) - t_log_var.mapv(|v| v.exp())).sum();
        (rec_loss + kl_loss) / y.nrows() as f64
    }

    /// Compute gradients of the VAE loss with respect to y_pred, t_log_var, and t_mean.
    pub fn grad(
        &self,
        y: &Array2<f64>,
        y_pred: &Array2<f64>,
        t_mean: &Array2<f64>,
        t_log_var: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let n = y.nrows() as f64;
        let eps = f64::EPSILON;
        let y_pred_clipped = y_pred.mapv(|v| v.max(eps).min(1.0 - eps));
        let dy_pred = -y / (n * &y_pred_clipped) - (y - 1.0) / (n - n * &y_pred_clipped);
        let d_log_var = (&t_log_var.mapv(|v| v.exp()) - 1.0) / (2.0 * n);
        let d_mean = t_mean / n;
        (dy_pred, d_log_var, d_mean)
    }
}

/// Create a loss function by name string.
pub fn create_loss(name: &str) -> Box<dyn Loss> {
    match name {
        "SquaredError" => Box::new(SquaredError::new()),
        "CrossEntropy" => Box::new(CrossEntropy::new()),
        _ => Box::new(SquaredError::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_squared_error() {
        let loss_fn = SquaredError::new();
        let y = array![[1.0, 0.0, 0.0]];
        let y_pred = array![[0.9, 0.1, 0.0]];
        let loss = loss_fn.loss(&y, &y_pred);
        // 0.5 * ||diff||² / n = 0.5 * (0.01 + 0.01 + 0) / 3 = 0.01/3
        assert!(loss > 0.0);
        assert!(loss < 0.01);
    }

    #[test]
    fn test_cross_entropy() {
        let loss_fn = CrossEntropy::new();
        let y = array![[1.0, 0.0], [0.0, 1.0]];
        let y_pred = array![[0.9, 0.1], [0.1, 0.9]];
        let loss = loss_fn.loss(&y, &y_pred);
        assert!(loss > 0.0);
        assert!(loss < 1.0);
    }
}
