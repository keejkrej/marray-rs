use ndarray::Array2;

use super::preprocessing::{MinMaxScaler, StandardScaler};

pub trait Transformer {
    fn fit(&mut self, x: &Array2<f64>);
    fn transform(&self, x: &Array2<f64>) -> Array2<f64>;
    fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x)
    }
}

pub struct StandardScalerWrapper {
    scaler: StandardScaler,
}

impl StandardScalerWrapper {
    pub fn new() -> Self {
        Self {
            scaler: StandardScaler::new(),
        }
    }
}

impl Default for StandardScalerWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for StandardScalerWrapper {
    fn fit(&mut self, x: &Array2<f64>) {
        self.scaler.fit(x);
    }

    fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        self.scaler.transform(x)
    }

    fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.scaler.fit_transform(x)
    }
}

pub struct MinMaxScalerWrapper {
    scaler: MinMaxScaler,
}

impl MinMaxScalerWrapper {
    pub fn new() -> Self {
        Self {
            scaler: MinMaxScaler::new(),
        }
    }
}

impl Default for MinMaxScalerWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for MinMaxScalerWrapper {
    fn fit(&mut self, x: &Array2<f64>) {
        self.scaler.fit(x);
    }

    fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        self.scaler.transform(x)
    }

    fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.scaler.fit_transform(x)
    }
}

pub struct Pipeline {
    steps: Vec<Box<dyn Transformer>>,
}

impl Pipeline {
    pub fn new(steps: Vec<Box<dyn Transformer>>) -> Self {
        Self { steps }
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut current = x.clone();
        for step in self.steps.iter_mut() {
            current = step.fit_transform(&current);
        }
        current
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut current = x.clone();
        for step in self.steps.iter() {
            current = step.transform(&current);
        }
        current
    }
}

pub fn make_pipeline(steps: Vec<Box<dyn Transformer>>) -> Pipeline {
    Pipeline::new(steps)
}
