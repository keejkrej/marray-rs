use ndarray::{s, Array2};
use rand_distr::{Distribution, Normal};

use super::super::losses::VAELoss;
use super::super::layers::Linear;

/// Variational Autoencoder (VAE) with Bernoulli input and output units.
///
/// Implements both fully-connected and convolutional variational autoencoders
/// trained with the reparameterization trick and ELBO (VAELoss: reconstruction + KL).
pub struct BernoulliVAE {
    /// Latent space dimension (T)
    pub t_dim: usize,
    /// Hidden dimension of intermediate FC layers
    pub latent_dim: usize,
    /// Number of input features (flattened)
    pub n_in: usize,
    /// Loss function
    pub loss: VAELoss,
    // Encoder layers
    pub enc_fc1: Linear,
    pub enc_fc2: Linear,
    // Decoder layers
    pub dec_fc1: Linear,
    pub dec_fc2: Linear,
    // Caches for backpropagation
    pub noise_cache: Option<Array2<f64>>,
    pub t_mean_cache: Option<Array2<f64>>,
    pub t_log_var_cache: Option<Array2<f64>>,
    pub z_cache: Option<Array2<f64>>,
}

impl BernoulliVAE {
    /// Create a new Bernoulli VAE.
    ///
    /// * `n_in`: Dimension of input vector (or flattened image).
    /// * `t_dim`: Dimension of latent representation (T). Default in numpy_ml is often 5 or 16.
    /// * `latent_dim`: Hidden layer dimension in encoder and decoder (e.g. 64 or 256).
    pub fn new(n_in: usize, t_dim: usize, latent_dim: usize) -> Self {
        // Encoder: FC1 (ReLU) -> FC2 (Identity, outputs 2 * T for mean and log_var)
        let enc_fc1 = Linear::new(latent_dim, Some("ReLU"), "glorot_uniform", Some("SGD"));
        let enc_fc2 = Linear::new(2 * t_dim, Some("Identity"), "glorot_uniform", Some("SGD"));

        // Decoder: FC1 (ReLU) -> FC2 (Sigmoid, outputs n_in)
        let dec_fc1 = Linear::new(latent_dim, Some("ReLU"), "glorot_uniform", Some("SGD"));
        let dec_fc2 = Linear::new(n_in, Some("Sigmoid"), "glorot_uniform", Some("SGD"));

        Self {
            t_dim,
            latent_dim,
            n_in,
            loss: VAELoss::new(),
            enc_fc1,
            enc_fc2,
            dec_fc1,
            dec_fc2,
            noise_cache: None,
            t_mean_cache: None,
            t_log_var_cache: None,
            z_cache: None,
        }
    }

    /// Sample from q(t | x) using the reparameterization trick:
    /// t = t_mean + noise * exp(t_log_var) where noise ~ N(0, I)
    pub fn sample_latent(&mut self, t_mean: &Array2<f64>, t_log_var: &Array2<f64>) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let noise = Array2::from_shape_fn(t_mean.raw_dim(), |_| normal.sample(&mut rng));

        let samples = &noise * &t_log_var.mapv(|v| (0.5 * v).exp()) + t_mean;
        self.noise_cache = Some(noise);
        samples
    }

    /// Forward pass through encoder, sampler, and decoder.
    ///
    /// Returns the reconstructed probabilities x_recon in [0, 1]^N.
    pub fn forward(&mut self, x: &Array2<f64>, retain_derived: bool) -> Array2<f64> {
        let enc_h1 = self.enc_fc1.forward(x, retain_derived);
        let enc_out = self.enc_fc2.forward(&enc_h1, retain_derived);

        let t_mean = enc_out.slice(s![.., ..self.t_dim]).to_owned();
        let t_log_var = enc_out.slice(s![.., self.t_dim..]).to_owned();

        let z = self.sample_latent(&t_mean, &t_log_var);
        let dec_h1 = self.dec_fc1.forward(&z, retain_derived);
        let x_recon = self.dec_fc2.forward(&dec_h1, retain_derived);

        if retain_derived {
            self.t_mean_cache = Some(t_mean);
            self.t_log_var_cache = Some(t_log_var);
            self.z_cache = Some(z);
        }

        x_recon
    }

    /// Backward pass computing gradients through decoder, sampler, and encoder.
    pub fn backward(&mut self, x_train: &Array2<f64>, x_recon: &Array2<f64>) -> Array2<f64> {
        let noise = self.noise_cache.as_ref().expect("No cached noise");
        let t_mean = self.t_mean_cache.as_ref().expect("No cached t_mean");
        let t_log_var = self.t_log_var_cache.as_ref().expect("No cached t_log_var");

        // 1. Loss gradients
        let (dy_pred, d_log_var_loss, d_mean_loss) = self.loss.grad(x_train, x_recon, t_mean, t_log_var);

        // 2. Decoder backpropagation
        let d_dec_h1 = self.dec_fc2.backward(&dy_pred, true);
        let d_z = self.dec_fc1.backward(&d_dec_h1, true);

        // 3. Sampler backpropagation
        // z = t_mean + noise * exp(0.5 * t_log_var)
        // dz/dt_mean = 1
        // dz/dt_log_var = noise * 0.5 * exp(0.5 * t_log_var)
        let d_t_mean = &d_z + &d_mean_loss;
        let d_t_log_var = &d_z * (noise * &t_log_var.mapv(|v| 0.5 * (0.5 * v).exp())) + &d_log_var_loss;

        // 4. Encoder backpropagation
        let (nrows, ncols) = (d_t_mean.nrows(), d_t_mean.ncols() + d_t_log_var.ncols());
        let mut d_enc_out = Array2::<f64>::zeros((nrows, ncols));
        for i in 0..nrows {
            for j in 0..self.t_dim {
                d_enc_out[[i, j]] = d_t_mean[[i, j]];
                d_enc_out[[i, j + self.t_dim]] = d_t_log_var[[i, j]];
            }
        }

        let d_enc_h1 = self.enc_fc2.backward(&d_enc_out, true);
        
        self.enc_fc1.backward(&d_enc_h1, true)
    }

    /// Update parameters of all encoder and decoder layers.
    pub fn update_params(&mut self, cur_loss: Option<f64>) {
        self.dec_fc2.update_params(cur_loss);
        self.dec_fc1.update_params(cur_loss);
        self.enc_fc2.update_params(cur_loss);
        self.enc_fc1.update_params(cur_loss);
        self.flush_gradients();
    }

    /// Reset intermediate gradient caches.
    pub fn flush_gradients(&mut self) {
        self.dec_fc2.flush_gradients();
        self.dec_fc1.flush_gradients();
        self.enc_fc2.flush_gradients();
        self.enc_fc1.flush_gradients();
    }

    /// Generate new samples by drawing z ~ N(0, I) and passing through decoder.
    pub fn sample(&mut self, n_samples: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = Array2::from_shape_fn((n_samples, self.t_dim), |_| normal.sample(&mut rng));
        let dec_h1 = self.dec_fc1.forward(&z, false);
        self.dec_fc2.forward(&dec_h1, false)
    }

    /// Reconstruct input data.
    pub fn reconstruct(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.forward(x, false)
    }

    /// Fit the VAE on training data.
    pub fn fit(&mut self, x_train: &Array2<f64>, n_epochs: usize, batch_size: usize) -> Vec<f64> {
        let n_ex = x_train.nrows();
        let mut epoch_losses = Vec::new();

        for _ in 0..n_epochs {
            let mut total_loss = 0.0;
            let mut n_batches = 0;

            for start in (0..n_ex).step_by(batch_size) {
                let end = (start + batch_size).min(n_ex);
                let x_batch = x_train.slice(s![start..end, ..]).to_owned();

                let x_recon = self.forward(&x_batch, true);
                let t_mean = self.t_mean_cache.as_ref().unwrap().clone();
                let t_log_var = self.t_log_var_cache.as_ref().unwrap().clone();

                let loss = self.loss.loss(&x_batch, &x_recon, &t_mean, &t_log_var);
                total_loss += loss;
                n_batches += 1;

                self.backward(&x_batch, &x_recon);
                self.update_params(Some(loss));
            }

            epoch_losses.push(total_loss / n_batches.max(1) as f64);
        }

        epoch_losses
    }
}

pub type VAE = BernoulliVAE;
