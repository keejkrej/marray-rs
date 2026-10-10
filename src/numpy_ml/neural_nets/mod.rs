//! Neural network building blocks: activations, losses, optimizers,
//! schedulers, initializers, and layers.

pub mod activations;
pub mod initializers;
pub mod layers;
pub mod losses;
pub mod models;
pub mod optimizers;
pub mod schedulers;
pub mod utils;

#[cfg(test)]
mod tests {
    use super::activations::*;
    use super::initializers::*;
    use super::layers::*;
    use super::losses::*;
    use super::optimizers::*;
    use super::schedulers::*;
    use super::utils;
    use ndarray::{array, Array2};

    #[test]
    fn test_activation_trait() {
        let act = Sigmoid::new();
        let z = array![[0.0, 1.0, -1.0]];
        let y = act.fn_(&z);
        assert!((y[[0, 0]] - 0.5).abs() < 1e-10);

        let grad = act.grad(&z);
        assert!(grad.iter().all(|&v| v > 0.0 && v <= 0.25));
    }

    #[test]
    fn test_loss_trait() {
        let loss_fn = SquaredError::new();
        let y = array![[1.0, 0.0]];
        let y_pred = array![[0.9, 0.1]];
        let loss = loss_fn.loss(&y, &y_pred);
        // 0.5 * ((-0.1)^2 + (0.1)^2) / 2 = 0.5 * 0.02 / 2 = 0.005
        assert!((loss - 0.005).abs() < 1e-6);
    }

    #[test]
    fn test_optimizer_trait() {
        let mut opt = SGD::new(0.1, 0.0, None);
        let param = array![[1.0]];
        let grad = array![[0.1]];
        let new_param = opt.update(&param, &grad, "w", None);
        assert!((new_param[[0, 0]] - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_trait() {
        let s = ConstantScheduler::new(0.01);
        assert_eq!(s.learning_rate(0, None), 0.01);

        let exp = ExponentialScheduler::new(0.1, 100, false, 0.1);
        let lr = exp.learning_rate(100, None);
        assert!((lr - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_glorot_initialization() {
        let w = glorot_uniform(128, 64, 1.0);
        assert_eq!(w.dim(), (128, 64));
        // Check roughly bounded
        let max_abs = w.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_abs < 1.0);
    }

    #[test]
    fn test_linear_layer() {
        let mut layer = Linear::new(4, Some("Tanh"), "glorot_uniform", Some("SGD"));
        let x = array![[1.0, 2.0, 3.0]];
        let y = layer.forward(&x, true);
        assert_eq!(y.dim(), (1, 4));

        let dldy = Array2::ones(y.raw_dim());
        let dx = layer.backward(&dldy, true);
        assert_eq!(dx.dim(), x.dim());
    }

    #[test]
    fn test_lstm_cell() {
        let mut cell = LSTMCell::new(4, Some("Tanh"), Some("Sigmoid"), "glorot_uniform", Some("SGD"));
        let xt = array![[1.0, 2.0, 3.0]];
        let (at, ct) = cell.forward(&xt);
        assert_eq!(at.dim(), (1, 4));
        assert_eq!(ct.dim(), (1, 4));
    }

    #[test]
    fn test_weight_initializers() {
        let w = he_normal(100, 50);
        assert_eq!(w.dim(), (100, 50));
        let mean: f64 = w.iter().sum::<f64>() / w.len() as f64;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_softmax() {
        let x = array![[1.0, 2.0, 3.0]];
        let y = utils::softmax(&x);
        let sum: f64 = y.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
