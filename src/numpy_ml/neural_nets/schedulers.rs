use std::f64;

/// Trait for all learning rate schedulers.
pub trait Scheduler {
    /// Compute the learning rate for the current step.
    fn learning_rate(&self, step: usize, cur_loss: Option<f64>) -> f64;
    /// Return the name of the scheduler.
    fn name(&self) -> &str;
}

/// Constant learning rate scheduler.
pub struct ConstantScheduler {
    lr: f64,
}

impl ConstantScheduler {
    pub fn new(lr: f64) -> Self {
        ConstantScheduler { lr }
    }
}

impl Scheduler for ConstantScheduler {
    fn learning_rate(&self, _step: usize, _cur_loss: Option<f64>) -> f64 {
        self.lr
    }

    fn name(&self) -> &str {
        "ConstantScheduler"
    }
}

/// Exponential decay scheduler.
pub struct ExponentialScheduler {
    initial_lr: f64,
    stage_length: usize,
    staircase: bool,
    decay: f64,
}

impl ExponentialScheduler {
    pub fn new(initial_lr: f64, stage_length: usize, staircase: bool, decay: f64) -> Self {
        ExponentialScheduler {
            initial_lr,
            stage_length,
            staircase,
            decay,
        }
    }
}

impl Scheduler for ExponentialScheduler {
    fn learning_rate(&self, step: usize, _cur_loss: Option<f64>) -> f64 {
        let cur_stage = if self.staircase {
            (step / self.stage_length) as f64
        } else {
            step as f64 / self.stage_length as f64
        };
        self.initial_lr * self.decay.powf(cur_stage)
    }

    fn name(&self) -> &str {
        "ExponentialScheduler"
    }
}

/// Noam scheduler (from "Attention is All You Need").
pub struct NoamScheduler {
    model_dim: f64,
    scale_factor: f64,
    warmup_steps: f64,
}

impl NoamScheduler {
    pub fn new(model_dim: f64, scale_factor: f64, warmup_steps: f64) -> Self {
        NoamScheduler {
            model_dim,
            scale_factor,
            warmup_steps,
        }
    }
}

impl Scheduler for NoamScheduler {
    fn learning_rate(&self, step: usize, _cur_loss: Option<f64>) -> f64 {
        let step = step as f64;
        if step == 0.0 {
            return 0.0;
        }
        let new_lr = self.model_dim.powf(-0.5)
            * step.powf(-0.5).min(step * self.warmup_steps.powf(-1.5));
        self.scale_factor * new_lr
    }

    fn name(&self) -> &str {
        "NoamScheduler"
    }
}

/// King scheduler (Davis King / DLib style).
///
/// Monitors loss history and decays LR when loss stops decreasing.
pub struct KingScheduler {
    #[allow(dead_code)]
    initial_lr: f64,
    patience: usize,
    decay: f64,
    current_lr: f64,
    loss_history: Vec<f64>,
}

impl KingScheduler {
    pub fn new(initial_lr: f64, patience: usize, decay: f64) -> Self {
        KingScheduler {
            initial_lr,
            patience,
            decay,
            current_lr: initial_lr,
            loss_history: Vec::new(),
        }
    }

    fn _steps_without_decrease(&self) -> usize {
        let lh = &self.loss_history;
        let n = lh.len();
        if n < 3 {
            return 0;
        }

        // Simple linear regression on loss history to estimate slope
        let start = if n > self.patience + 1 {
            n - self.patience - 1
        } else {
            0
        };
        let subset: Vec<f64> = lh[start..].to_vec();
        let nn = subset.len() as f64;

        if nn < 2.0 {
            return 0;
        }

        let x_mean = (nn - 1.0) / 2.0;
        let y_mean = subset.iter().sum::<f64>() / nn;

        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in subset.iter().enumerate() {
            let x = i as f64;
            num += (x - x_mean) * (y - y_mean);
            den += (x - x_mean) * (x - x_mean);
        }

        if den == 0.0 {
            return 0;
        }

        let slope = num / den;

        // Heuristic: if slope >= 0, loss is not decreasing
        if slope >= 0.0 {
            n - start
        } else {
            0
        }
    }
}

impl Scheduler for KingScheduler {
    fn learning_rate(&self, _step: usize, _cur_loss: Option<f64>) -> f64 {
        self.current_lr
    }

    fn name(&self) -> &str {
        "KingScheduler"
    }
}

impl KingScheduler {
    /// Update the scheduler with a new loss value and return the current learning rate.
    pub fn update_with_loss(&mut self, cur_loss: f64) -> f64 {
        self.loss_history.push(cur_loss);

        if self.loss_history.len() < self.patience {
            return self.current_lr;
        }

        // Keep at most 1.1 * (patience + 1) entries
        let max_history = (1.1 * (self.patience + 1) as f64).ceil() as usize;
        if self.loss_history.len() > max_history {
            let excess = self.loss_history.len() - max_history;
            self.loss_history.drain(..excess);
        }

        let steps = self._steps_without_decrease();
        if steps > self.patience {
            self.current_lr *= self.decay;
        }

        self.current_lr
    }
}

/// Create a scheduler by name string.
pub fn create_scheduler(name: &str, lr: f64) -> Box<dyn Scheduler> {
    match name {
        "ConstantScheduler" => Box::new(ConstantScheduler::new(lr)),
        "ExponentialScheduler" => Box::new(ExponentialScheduler::new(lr, 500, false, 0.1)),
        "NoamScheduler" => Box::new(NoamScheduler::new(512.0, 1.0, 4000.0)),
        _ => Box::new(ConstantScheduler::new(lr)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let s = ConstantScheduler::new(0.01);
        assert_eq!(s.learning_rate(0, None), 0.01);
        assert_eq!(s.learning_rate(100, None), 0.01);
    }

    #[test]
    fn test_exponential_scheduler() {
        let s = ExponentialScheduler::new(0.1, 100, false, 0.1);
        let lr = s.learning_rate(0, None);
        assert!((lr - 0.1).abs() < 1e-10);
        let lr = s.learning_rate(100, None);
        assert!((lr - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_noam_scheduler() {
        let s = NoamScheduler::new(512.0, 1.0, 4000.0);
        let lr = s.learning_rate(1, None);
        assert!(lr > 0.0);
        let lr2 = s.learning_rate(100, None);
        assert!(lr2 > lr);
    }
}
