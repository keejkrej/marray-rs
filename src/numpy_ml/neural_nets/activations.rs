use ndarray::Array2;
use std::f64;

/// Trait for all activation functions.
pub trait Activation {
    /// Apply the activation function to an input array.
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64>;
    /// Compute the first derivative of the activation function.
    fn grad(&self, x: &Array2<f64>) -> Array2<f64>;
    /// Compute the second derivative of the activation function.
    fn grad2(&self, x: &Array2<f64>) -> Array2<f64>;
    /// Return the name of the activation function.
    fn name(&self) -> &str;
}

/// Logistic sigmoid activation: σ(x) = 1 / (1 + exp(-x))
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Sigmoid {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        let s = self.fn_(x);
        s.clone() * &(1.0 - &s)
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        let s = self.fn_(x);
        &s * &(1.0 - &s) * &(1.0 - 2.0 * &s)
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// ReLU activation: max(0, x)
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for ReLU {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| v.max(0.0))
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        Array2::zeros(x.raw_dim())
    }

    fn name(&self) -> &str {
        "ReLU"
    }
}

/// Leaky ReLU activation: x if x > 0, alpha * x otherwise
pub struct LeakyReLU {
    pub alpha: f64,
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        LeakyReLU { alpha }
    }
}

impl Activation for LeakyReLU {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| if v > 0.0 { v } else { self.alpha * v })
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { self.alpha })
    }

    fn grad2(&self, _x: &Array2<f64>) -> Array2<f64> {
        Array2::zeros(_x.raw_dim())
    }

    fn name(&self) -> &str {
        "LeakyReLU"
    }
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub struct GELU {
    pub approximate: bool,
}

impl GELU {
    pub fn new(approximate: bool) -> Self {
        GELU { approximate }
    }
}

impl Activation for GELU {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        let pi = f64::consts::PI;
        if self.approximate {
            z.mapv(|v| {
                0.5 * v * (1.0 + ((2.0 / pi).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
            })
        } else {
            z.mapv(|v| {
                0.5 * v * (1.0 + erf(v / 2.0_f64.sqrt()))
            })
        }
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        let pi = f64::consts::PI;
        let sqrt_2 = 2.0_f64.sqrt();
        let erf_prime = |v: f64| -> f64 { (2.0 / pi.sqrt()) * (-v * v).exp() };

        if self.approximate {
            x.mapv(|v| {
                let s = v / sqrt_2;
                let approx = ((2.0 / pi).sqrt() * (v + 0.044715 * v.powi(3))).tanh();
                0.5 + 0.5 * approx + (0.5 * v * erf_prime(s)) / sqrt_2
            })
        } else {
            x.mapv(|v| {
                let s = v / sqrt_2;
                0.5 + 0.5 * erf(s) + (0.5 * v * erf_prime(s)) / sqrt_2
            })
        }
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        let sqrt_2 = 2.0_f64.sqrt();
        let pi = f64::consts::PI;
        let erf_prime = |v: f64| -> f64 { (2.0 / pi.sqrt()) * (-v * v).exp() };
        let erf_prime2 = |v: f64| -> f64 { -4.0 * v * (-v * v).exp() / pi.sqrt() };

        x.mapv(|v| {
            let s = v / sqrt_2;
            (1.0 / (2.0 * sqrt_2)) * (erf_prime(s) + erf_prime2(s) / sqrt_2)
        })
    }

    fn name(&self) -> &str {
        "GELU"
    }
}

/// Hyperbolic tangent activation
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Tanh {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| v.tanh())
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| {
            let t = v.tanh();
            1.0 - t * t
        })
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| {
            let t = v.tanh();
            let g = 1.0 - t * t;
            -2.0 * t * g
        })
    }

    fn name(&self) -> &str {
        "Tanh"
    }
}

/// Affine activation: slope * x + intercept
pub struct Affine {
    pub slope: f64,
    pub intercept: f64,
}

impl Affine {
    pub fn new(slope: f64, intercept: f64) -> Self {
        Affine { slope, intercept }
    }
}

impl Activation for Affine {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| self.slope * v + self.intercept)
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        Array2::from_elem(x.raw_dim(), self.slope)
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        Array2::zeros(x.raw_dim())
    }

    fn name(&self) -> &str {
        "Affine"
    }
}

/// Identity activation (Affine with slope=1, intercept=0)
pub struct Identity;

impl Identity {
    pub fn new() -> Self {
        Identity
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Identity {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.clone()
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        Array2::from_elem(x.raw_dim(), 1.0)
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        Array2::zeros(x.raw_dim())
    }

    fn name(&self) -> &str {
        "Identity"
    }
}

/// ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise
pub struct ELU {
    pub alpha: f64,
}

impl ELU {
    pub fn new(alpha: f64) -> Self {
        ELU { alpha }
    }
}

impl Activation for ELU {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| if v > 0.0 { v } else { self.alpha * (v.exp() - 1.0) })
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { self.alpha * v.exp() })
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v >= 0.0 { 0.0 } else { self.alpha * v.exp() })
    }

    fn name(&self) -> &str {
        "ELU"
    }
}

/// Exponential activation: exp(x)
pub struct Exponential;

impl Exponential {
    pub fn new() -> Self {
        Exponential
    }
}

impl Default for Exponential {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Exponential {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| v.exp())
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.exp())
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.exp())
    }

    fn name(&self) -> &str {
        "Exponential"
    }
}

/// SELU activation: scale * ELU(x, alpha)
pub struct SELU {
    alpha: f64,
    scale: f64,
}

impl SELU {
    pub fn new() -> Self {
        SELU {
            alpha: 1.673_263_242_354_377_2,
            scale: 1.050_700_987_355_480_5,
        }
    }
}

impl Default for SELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for SELU {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        let elu = ELU::new(self.alpha);
        let e = elu.fn_(z);
        e.mapv(|v| self.scale * v)
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| {
            if v >= 0.0 {
                self.scale
            } else {
                self.scale * self.alpha * v.exp()
            }
        })
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| {
            if v > 0.0 {
                0.0
            } else {
                self.scale * self.alpha * v.exp()
            }
        })
    }

    fn name(&self) -> &str {
        "SELU"
    }
}

/// Hard Sigmoid: clip(0.2 * x + 0.5, 0, 1)
pub struct HardSigmoid;

impl HardSigmoid {
    pub fn new() -> Self {
        HardSigmoid
    }
}

impl Default for HardSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for HardSigmoid {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| (0.2 * v + 0.5).max(0.0).min(1.0))
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if (-2.5..=2.5).contains(&v) { 0.2 } else { 0.0 })
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        Array2::zeros(x.raw_dim())
    }

    fn name(&self) -> &str {
        "HardSigmoid"
    }
}

/// SoftPlus activation: log(1 + exp(x))
pub struct SoftPlus;

impl SoftPlus {
    pub fn new() -> Self {
        SoftPlus
    }
}

impl Default for SoftPlus {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for SoftPlus {
    fn fn_(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| (v.exp() + 1.0).ln())
    }

    fn grad(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| {
            let e = v.exp();
            e / (e + 1.0)
        })
    }

    fn grad2(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| {
            let e = v.exp();
            e / ((e + 1.0) * (e + 1.0))
        })
    }

    fn name(&self) -> &str {
        "SoftPlus"
    }
}

/// Error function approximation using Abramowitz and Stegun formula
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + 0.3275911 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = 0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3
        - 1.453152027 * t4
        + 1.061405429 * t5;

    sign * (1.0 - poly * (-x * x).exp())
}

/// Create an activation function by name string.
pub fn create_activation(name: &str) -> Box<dyn Activation> {
    match name {
        "Sigmoid" => Box::new(Sigmoid::new()),
        "ReLU" => Box::new(ReLU::new()),
        "Tanh" => Box::new(Tanh::new()),
        "Identity" => Box::new(Identity::new()),
        "Exponential" => Box::new(Exponential::new()),
        "SELU" => Box::new(SELU::new()),
        "HardSigmoid" => Box::new(HardSigmoid::new()),
        "SoftPlus" => Box::new(SoftPlus::new()),
        "GELU" => Box::new(GELU::new(true)),
        "ELU" => Box::new(ELU::new(1.0)),
        _ => {
            if name.starts_with("LeakyReLU") {
                Box::new(LeakyReLU::new(0.3))
            } else if name.starts_with("Affine") {
                Box::new(Affine::new(1.0, 0.0))
            } else {
                Box::new(Identity::new())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        let act = Sigmoid::new();
        let z = array![[0.0, 1.0, -1.0]];
        let y = act.fn_(&z);
        assert!((y[[0, 0]] - 0.5).abs() < 1e-10);
        assert!((y[[0, 1]] - 0.7310585786300049).abs() < 1e-6);
        assert!((y[[0, 2]] - 0.2689414213699951).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let act = ReLU::new();
        let z = array![[-1.0, 0.0, 1.0, 2.0]];
        let y = act.fn_(&z);
        assert_eq!(y, array![[0.0, 0.0, 1.0, 2.0]]);
    }

    #[test]
    fn test_tanh() {
        let act = Tanh::new();
        let z = array![[0.0]];
        let y = act.fn_(&z);
        assert!((y[[0, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_identity() {
        let act = Identity::new();
        let z = array![[1.0, 2.0, 3.0]];
        assert_eq!(act.fn_(&z), z);
    }

    #[test]
    fn test_leaky_relu() {
        let act = LeakyReLU::new(0.01);
        let z = array![[-1.0, 1.0]];
        let y = act.fn_(&z);
        assert!((y[[0, 0]] - (-0.01)).abs() < 1e-10);
        assert!((y[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_elu() {
        let act = ELU::new(1.0);
        let z = array![[-1.0, 1.0]];
        let y = act.fn_(&z);
        assert!((y[[0, 0]] - (-0.6321205588285577)).abs() < 1e-6);
        assert!((y[[0, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softplus() {
        let act = SoftPlus::new();
        let z = array![[0.0]];
        let y = act.fn_(&z);
        assert!((y[[0, 0]] - 0.6931471805599453).abs() < 1e-6);
    }
}
