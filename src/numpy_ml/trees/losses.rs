use ndarray::Array1;

/// Base estimator that predicts the empirical class probability or mean.
pub trait BaseEstimator: Send + Sync {
    fn fit(&mut self, y: &Array1<f64>);
    fn predict(&self, n_samples: usize) -> Array1<f64>;
}

pub struct ClassProbEstimator {
    class_prob: f64,
}

impl ClassProbEstimator {
    pub fn new() -> Self {
        Self { class_prob: 0.0 }
    }
}

impl Default for ClassProbEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseEstimator for ClassProbEstimator {
    fn fit(&mut self, y: &Array1<f64>) {
        let n = y.len().max(1) as f64;
        self.class_prob = y.sum() / n;
    }

    fn predict(&self, n_samples: usize) -> Array1<f64> {
        Array1::from_elem(n_samples, self.class_prob)
    }
}

pub struct MeanBaseEstimator {
    avg: f64,
}

impl MeanBaseEstimator {
    pub fn new() -> Self {
        Self { avg: 0.0 }
    }
}

impl Default for MeanBaseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseEstimator for MeanBaseEstimator {
    fn fit(&mut self, y: &Array1<f64>) {
        self.avg = y.mean().unwrap_or(0.0);
    }

    fn predict(&self, n_samples: usize) -> Array1<f64> {
        Array1::from_elem(n_samples, self.avg)
    }
}

/// Loss function trait for tree-based models like GBDT.
pub trait TreeLoss: Send + Sync {
    fn loss(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> f64;
    fn grad(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> Array1<f64>;
    fn base_estimator(&self) -> Box<dyn BaseEstimator>;
}

/// Mean squared error loss for regression trees.
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeLoss for MSELoss {
    fn loss(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let diff = y - y_pred;
        diff.iter().map(|&v| v * v).sum::<f64>() / y.len().max(1) as f64
    }

    fn grad(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> Array1<f64> {
        let n = y.len().max(1) as f64;
        let diff = y - y_pred;
        &diff * (-2.0 / n)
    }

    fn base_estimator(&self) -> Box<dyn BaseEstimator> {
        Box::new(MeanBaseEstimator::new())
    }
}

/// Cross-entropy loss for binary classification trees.
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeLoss for CrossEntropyLoss {
    fn loss(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let eps = 1e-15;
        let mut total = 0.0;
        for i in 0..y.len() {
            let p = y_pred[i].clamp(eps, 1.0 - eps);
            total += -y[i] * p.ln() - (1.0 - y[i]) * (1.0 - p).ln();
        }
        total / y.len().max(1) as f64
    }

    fn grad(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> Array1<f64> {
        let eps = 1e-15;
        Array1::from_shape_fn(y.len(), |i| {
            let p = y_pred[i].clamp(eps, 1.0 - eps);
            -(y[i] / p - (1.0 - y[i]) / (1.0 - p))
        })
    }

    fn base_estimator(&self) -> Box<dyn BaseEstimator> {
        Box::new(ClassProbEstimator::new())
    }
}
