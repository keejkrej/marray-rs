#![allow(dead_code)]

use ndarray::{s, Array1, Array2, Array3, Array4};
use rand::Rng;

use super::activations::{create_activation, Activation};
use super::initializers;
use super::optimizers::{create_optimizer, Optimizer};

/// A fully connected (dense/linear) layer.
///
/// Forward: Y = act_fn(X @ W + b)
pub struct Linear {
    pub n_in: usize,
    pub n_out: usize,
    pub trainable: bool,
    pub parameters: Option<LinearParams>,
    pub gradients: Option<LinearGrads>,
    act_fn: Box<dyn Activation>,
    optimizer: Box<dyn Optimizer>,
    init: String,
    x_cache: Vec<Array2<f64>>,
    z_cache: Vec<Array2<f64>>,
}

pub struct LinearParams {
    pub w: Array2<f64>,
    pub b: Array2<f64>,
}

pub struct LinearGrads {
    pub dw: Array2<f64>,
    pub db: Array2<f64>,
}

impl Linear {
    pub fn new(n_out: usize, act_fn: Option<&str>, init: &str, optimizer: Option<&str>) -> Self {
        Linear {
            n_in: 0,
            n_out,
            trainable: true,
            parameters: None,
            gradients: None,
            act_fn: create_activation(act_fn.unwrap_or("Identity")),
            optimizer: create_optimizer(optimizer.unwrap_or("SGD")),
            init: init.to_string(),
            x_cache: Vec::new(),
            z_cache: Vec::new(),
        }
    }

    fn init_params(&mut self, n_in: usize) {
        self.n_in = n_in;
        let gain = initializers::calc_glorot_gain(self.act_fn.name());
        let w = initializers::glorot_uniform(n_in, self.n_out, gain);
        let b = Array2::zeros((1, self.n_out));
        self.parameters = Some(LinearParams { w, b: b.clone() });
        self.gradients = Some(LinearGrads {
            dw: Array2::zeros((n_in, self.n_out)),
            db: Array2::zeros((1, self.n_out)),
        });
    }

    pub fn forward(&mut self, x: &Array2<f64>, retain_derived: bool) -> Array2<f64> {
        if self.parameters.is_none() {
            self.init_params(x.ncols());
        }

        let params = self.parameters.as_ref().unwrap();
        let z = x.dot(&params.w) + &params.b;
        let y = self.act_fn.fn_(&z);

        if retain_derived {
            self.x_cache.push(x.clone());
            self.z_cache.push(z);
        }

        y
    }

    pub fn backward(&mut self, dldy: &Array2<f64>, retain_grads: bool) -> Array2<f64> {
        assert!(self.trainable, "Layer is frozen");

        let x = self.x_cache.pop().unwrap();
        let z = self.z_cache.pop().unwrap();
        let params = self.parameters.as_ref().unwrap();

        let dz = dldy * &self.act_fn.grad(&z);
        let dw = x.t().dot(&dz);
        let db = dz.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        let dx = dz.dot(&params.w.t());

        if retain_grads {
            let grads = self.gradients.as_mut().unwrap();
            grads.dw += &dw;
            grads.db += &db;
        }

        dx
    }

    pub fn update_params(&mut self, cur_loss: Option<f64>) {
        if let (Some(params), Some(grads)) =
            (&self.parameters, &self.gradients)
        {
            let w = self.optimizer.update(&params.w, &grads.dw, "w", cur_loss);
            let b = self.optimizer.update(&params.b, &grads.db, "b", cur_loss);
            self.parameters = Some(LinearParams { w, b });
        }
    }

    pub fn flush_gradients(&mut self) {
        if let Some(ref params) = self.parameters
            && let Some(ref mut grads) = self.gradients {
                grads.dw = Array2::zeros(params.w.raw_dim());
                grads.db = Array2::zeros(params.b.raw_dim());
            }
        self.x_cache.clear();
        self.z_cache.clear();
    }

    pub fn freeze(&mut self) {
        self.trainable = false;
    }

    pub fn unfreeze(&mut self) {
        self.trainable = true;
    }
}

/// A vanilla (Elman) RNN cell.
///
/// A[t] = act_fn(Wax @ Xt + Waa @ A[t-1] + ba)
pub struct RNNCell {
    pub n_in: usize,
    pub n_out: usize,
    pub trainable: bool,
    parameters: Option<RNNParams>,
    gradients: Option<RNNGrads>,
    act_fn: Box<dyn Activation>,
    optimizer: Box<dyn Optimizer>,
    a_cache: Vec<Array2<f64>>,
    z_cache: Vec<Array2<f64>>,
    x_cache: Vec<Array2<f64>>,
    current_step: usize,
    dldx_acc: Option<Array2<f64>>,
}

struct RNNParams {
    wax: Array2<f64>,
    waa: Array2<f64>,
    ba: Array2<f64>,
}

struct RNNGrads {
    wax: Array2<f64>,
    waa: Array2<f64>,
    ba: Array2<f64>,
}

impl RNNCell {
    pub fn new(n_out: usize, act_fn: Option<&str>, _init: &str, _optimizer: Option<&str>) -> Self {
        RNNCell {
            n_in: 0,
            n_out,
            trainable: true,
            parameters: None,
            gradients: None,
            act_fn: create_activation(act_fn.unwrap_or("Tanh")),
            optimizer: create_optimizer(_optimizer.unwrap_or("SGD")),
            a_cache: Vec::new(),
            z_cache: Vec::new(),
            x_cache: Vec::new(),
            current_step: 0,
            dldx_acc: None,
        }
    }

    fn init_params(&mut self) {
        let gain = initializers::calc_glorot_gain(self.act_fn.name());
        let wax = initializers::glorot_uniform(self.n_in, self.n_out, gain);
        let waa = initializers::glorot_uniform(self.n_out, self.n_out, gain);
        let ba = Array2::zeros((1, self.n_out));

        self.parameters = Some(RNNParams {
            wax,
            waa,
            ba: ba.clone(),
        });
        self.gradients = Some(RNNGrads {
            wax: Array2::zeros((self.n_in, self.n_out)),
            waa: Array2::zeros((self.n_out, self.n_out)),
            ba,
        });
    }

    pub fn forward(&mut self, xt: &Array2<f64>) -> Array2<f64> {
        if self.parameters.is_none() {
            self.n_in = xt.ncols();
            self.init_params();
        }

        let params = self.parameters.as_ref().unwrap();
        let a_prev = if self.a_cache.is_empty() {
            Array2::zeros((xt.nrows(), self.n_out))
        } else {
            self.a_cache.last().unwrap().clone()
        };

        let z = xt.dot(&params.wax) + a_prev.dot(&params.waa) + &params.ba;
        let at = self.act_fn.fn_(&z);

        self.z_cache.push(z);
        self.a_cache.push(at.clone());
        self.x_cache.push(xt.clone());

        at
    }

    pub fn backward(&mut self, dldat: &Array2<f64>) -> Array2<f64> {
        assert!(self.trainable, "Layer is frozen");

        if self.current_step == 0 {
            self.current_step = self.a_cache.len() - 1;
        }

        let t = self.current_step;
        let params = self.parameters.as_ref().unwrap();

        let z_t = &self.z_cache[t];
        let a_prev = if t > 0 { &self.a_cache[t - 1] } else { &self.a_cache[0] };
        let x_t = &self.x_cache[t];

        let dldx_acc = self.dldx_acc.take().unwrap_or_else(|| {
            Array2::zeros((dldat.nrows(), self.n_out))
        });

        let da = dldat + &dldx_acc;
        let dz = &self.act_fn.grad(z_t) * &da;

        let dwax = x_t.t().dot(&dz);
        let dwaa = a_prev.t().dot(&dz);
        let dba = dz.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        let dx = dz.dot(&params.wax.t());
        let da_prev = dz.dot(&params.waa.t());

        self.dldx_acc = Some(da_prev);
        self.current_step = t.wrapping_sub(1);

        if let Some(ref mut grads) = self.gradients {
            grads.wax += &dwax;
            grads.waa += &dwaa;
            grads.ba += &dba;
        }

        dx
    }

    pub fn flush_gradients(&mut self) {
        if let Some(ref params) = self.parameters
            && let Some(ref mut grads) = self.gradients {
                grads.wax = Array2::zeros(params.wax.raw_dim());
                grads.waa = Array2::zeros(params.waa.raw_dim());
                grads.ba = Array2::zeros(params.ba.raw_dim());
            }
        self.a_cache.clear();
        self.z_cache.clear();
        self.x_cache.clear();
        self.current_step = 0;
        self.dldx_acc = None;
    }
}

/// A single LSTM cell.
///
/// Z[t] = [A[t-1], X[t]]
/// Gf[t] = gate_fn(Wf @ Z[t] + bf)  (forget gate)
/// Gu[t] = gate_fn(Wu @ Z[t] + bu)  (update gate)
/// Go[t] = gate_fn(Wo @ Z[t] + bo)  (output gate)
/// Cc[t] = act_fn(Wc @ Z[t] + bc)   (candidate)
/// C[t]  = Gf[t] * C[t-1] + Gu[t] * Cc[t]
/// A[t]  = Go[t] * act_fn(C[t])
pub struct LSTMCell {
    pub n_in: usize,
    pub n_out: usize,
    pub trainable: bool,
    parameters: Option<LSTMParams>,
    gradients: Option<LSTMGrads>,
    act_fn: Box<dyn Activation>,
    gate_fn: Box<dyn Activation>,
    optimizer: Box<dyn Optimizer>,
    a_cache: Vec<Array2<f64>>,
    c_cache: Vec<Array2<f64>>,
    gf_cache: Vec<Array2<f64>>,
    gu_cache: Vec<Array2<f64>>,
    go_cache: Vec<Array2<f64>>,
    cc_cache: Vec<Array2<f64>>,
    x_cache: Vec<Array2<f64>>,
    current_step: usize,
    dlda_acc: Option<Array2<f64>>,
    dldc_acc: Option<Array2<f64>>,
}

struct LSTMParams {
    wf: Array2<f64>,
    wu: Array2<f64>,
    wc: Array2<f64>,
    wo: Array2<f64>,
    bf: Array2<f64>,
    bu: Array2<f64>,
    bc: Array2<f64>,
    bo: Array2<f64>,
}

struct LSTMGrads {
    wf: Array2<f64>,
    wu: Array2<f64>,
    wc: Array2<f64>,
    wo: Array2<f64>,
    bf: Array2<f64>,
    bu: Array2<f64>,
    bc: Array2<f64>,
    bo: Array2<f64>,
}

impl LSTMCell {
    pub fn new(
        n_out: usize,
        act_fn: Option<&str>,
        gate_fn: Option<&str>,
        _init: &str,
        _optimizer: Option<&str>,
    ) -> Self {
        LSTMCell {
            n_in: 0,
            n_out,
            trainable: true,
            parameters: None,
            gradients: None,
            act_fn: create_activation(act_fn.unwrap_or("Tanh")),
            gate_fn: create_activation(gate_fn.unwrap_or("Sigmoid")),
            optimizer: create_optimizer(_optimizer.unwrap_or("SGD")),
            a_cache: Vec::new(),
            c_cache: Vec::new(),
            gf_cache: Vec::new(),
            gu_cache: Vec::new(),
            go_cache: Vec::new(),
            cc_cache: Vec::new(),
            x_cache: Vec::new(),
            current_step: 0,
            dlda_acc: None,
            dldc_acc: None,
        }
    }

    fn init_params(&mut self) {
        let input_dim = self.n_in + self.n_out;
        let gain = initializers::calc_glorot_gain(self.gate_fn.name());
        let gain_act = initializers::calc_glorot_gain(self.act_fn.name());

        let wf = initializers::glorot_uniform(input_dim, self.n_out, gain);
        let wu = initializers::glorot_uniform(input_dim, self.n_out, gain);
        let wc = initializers::glorot_uniform(input_dim, self.n_out, gain_act);
        let wo = initializers::glorot_uniform(input_dim, self.n_out, gain);

        let bf = Array2::zeros((1, self.n_out));
        let bu = Array2::zeros((1, self.n_out));
        let bc = Array2::zeros((1, self.n_out));
        let bo = Array2::zeros((1, self.n_out));

        self.parameters = Some(LSTMParams {
            wf: wf.clone(),
            wu: wu.clone(),
            wc: wc.clone(),
            wo: wo.clone(),
            bf: bf.clone(),
            bu: bu.clone(),
            bc: bc.clone(),
            bo: bo.clone(),
        });

        self.gradients = Some(LSTMGrads {
            wf: Array2::zeros(wf.raw_dim()),
            wu: Array2::zeros(wu.raw_dim()),
            wc: Array2::zeros(wc.raw_dim()),
            wo: Array2::zeros(wo.raw_dim()),
            bf: Array2::zeros(bf.raw_dim()),
            bu: Array2::zeros(bu.raw_dim()),
            bc: Array2::zeros(bc.raw_dim()),
            bo: Array2::zeros(bo.raw_dim()),
        });
    }

    pub fn forward(&mut self, xt: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        if self.parameters.is_none() {
            self.n_in = xt.ncols();
            self.init_params();
        }

        let params = self.parameters.as_ref().unwrap();

        let (a_prev, c_prev) = if self.a_cache.is_empty() {
            let zeros = Array2::zeros((xt.nrows(), self.n_out));
            (zeros.clone(), zeros)
        } else {
            (
                self.a_cache.last().unwrap().clone(),
                self.c_cache.last().unwrap().clone(),
            )
        };

        // Concatenate [A_prev, X_t] by horizontal stacking
        let (n_ex, _) = a_prev.dim();
        let mut z = Array2::zeros((n_ex, self.n_in + self.n_out));
        for i in 0..n_ex {
            for j in 0..self.n_out {
                z[[i, j]] = a_prev[[i, j]];
            }
            for j in 0..self.n_in {
                z[[i, self.n_out + j]] = xt[[i, j]];
            }
        }

        let gf = self.gate_fn.fn_(&(z.dot(&params.wf) + &params.bf));
        let gu = self.gate_fn.fn_(&(z.dot(&params.wu) + &params.bu));
        let go = self.gate_fn.fn_(&(z.dot(&params.wo) + &params.bo));
        let cc = self.act_fn.fn_(&(z.dot(&params.wc) + &params.bc));
        let c = &gf * &c_prev + &gu * &cc;
        let a = &go * &self.act_fn.fn_(&c);

        self.x_cache.push(xt.clone());
        self.a_cache.push(a.clone());
        self.c_cache.push(c.clone());
        self.gf_cache.push(gf);
        self.gu_cache.push(gu);
        self.go_cache.push(go);
        self.cc_cache.push(cc);

        (a, c)
    }

    pub fn backward(&mut self, dldat: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        assert!(self.trainable, "Layer is frozen");

        if self.current_step == 0 {
            self.current_step = self.a_cache.len() - 1;
        }

        let t = self.current_step;
        let params = self.parameters.as_ref().unwrap();

        let ct = &self.c_cache[t + 1];
        let c_prev = &self.c_cache[t];
        let zt = {
            let a_prev = if t > 0 { &self.a_cache[t - 1] } else { &self.a_cache[0] };
            let xt = &self.x_cache[t];
            let (n_ex, _) = a_prev.dim();
            let mut z = Array2::zeros((n_ex, self.n_in + self.n_out));
            for i in 0..n_ex {
                for j in 0..self.n_out {
                    z[[i, j]] = a_prev[[i, j]];
                }
                for j in 0..self.n_in {
                    z[[i, self.n_out + j]] = xt[[i, j]];
                }
            }
            z
        };

        let dlda_acc = self.dlda_acc.take().unwrap_or_else(|| {
            Array2::zeros((dldat.nrows(), self.n_out))
        });
        let dldc_acc = self.dldc_acc.take().unwrap_or_else(|| {
            Array2::zeros((dldat.nrows(), self.n_out))
        });

        let da = dldat + &dlda_acc;
        let dc = &dldc_acc + &da * &self.go_cache[t] * &self.act_fn.grad(ct);

        let _go = zt.dot(&params.wo) + &params.bo;
        let _gf = zt.dot(&params.wf) + &params.bf;
        let _gu = zt.dot(&params.wu) + &params.bu;
        let _gc = zt.dot(&params.wc) + &params.bc;

        let dgo = &da * &self.act_fn.fn_(ct) * &self.gate_fn.grad(&_go);
        let dgc = &dc * &self.gu_cache[t] * &self.act_fn.grad(&_gc);
        let dgu = &dc * &self.cc_cache[t] * &self.gate_fn.grad(&_gu);
        let dgf = &dc * c_prev * &self.gate_fn.grad(&_gf);

        let dz = dgf.dot(&params.wf.t())
            + dgu.dot(&params.wu.t())
            + dgc.dot(&params.wc.t())
            + dgo.dot(&params.wo.t());

        let n_out = self.n_out;
        let dx = dz.slice(s![.., n_out..]).to_owned();
        let da_prev = dz.slice(s![.., ..n_out]).to_owned();

        // Update parameter gradients
        if let Some(ref mut grads) = self.gradients {
            grads.wf += &zt.t().dot(&dgf);
            grads.wu += &zt.t().dot(&dgu);
            grads.wc += &zt.t().dot(&dgc);
            grads.wo += &zt.t().dot(&dgo);
            grads.bf += &dgf.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
            grads.bu += &dgu.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
            grads.bc += &dgc.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
            grads.bo += &dgo.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        }

        self.dlda_acc = Some(da_prev);
        self.dldc_acc = Some(&self.gf_cache[t] * &dc);
        self.current_step = t.wrapping_sub(1);

        (dx, da)
    }

    pub fn flush_gradients(&mut self) {
        self.a_cache.clear();
        self.c_cache.clear();
        self.gf_cache.clear();
        self.gu_cache.clear();
        self.go_cache.clear();
        self.cc_cache.clear();
        self.x_cache.clear();
        self.current_step = 0;
        self.dlda_acc = None;
        self.dldc_acc = None;
    }
}

/// A full RNN layer that processes a sequence.
pub struct RNNLayer {
    pub n_out: usize,
    cell: RNNCell,
}

impl RNNLayer {
    pub fn new(n_out: usize, act_fn: Option<&str>, init: &str, optimizer: Option<&str>) -> Self {
        RNNLayer {
            n_out,
            cell: RNNCell::new(n_out, act_fn, init, optimizer),
        }
    }

    /// Forward pass over a full sequence.
    /// X shape: (n_ex * n_t, n_in) - flattened sequence
    pub fn forward(&mut self, x: &Array2<f64>, n_t: usize) -> Vec<Array2<f64>> {
        let mut outputs = Vec::with_capacity(n_t);
        let n_ex = x.nrows() / n_t;
        for t in 0..n_t {
            let xt = x.slice(s![t * n_ex..(t + 1) * n_ex, ..]).to_owned();
            let yt = self.cell.forward(&xt);
            outputs.push(yt);
        }
        outputs
    }

    /// Backward pass over a full sequence.
    pub fn backward(&mut self, dldy: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let mut dxs = Vec::new();
        for t in (0..dldy.len()).rev() {
            let dxt = self.cell.backward(&dldy[t]);
            dxs.insert(0, dxt);
        }
        dxs
    }
}

/// A full LSTM layer that processes a sequence.
pub struct LSTMLayer {
    pub n_out: usize,
    cell: LSTMCell,
}

impl LSTMLayer {
    pub fn new(
        n_out: usize,
        act_fn: Option<&str>,
        gate_fn: Option<&str>,
        init: &str,
        optimizer: Option<&str>,
    ) -> Self {
        LSTMLayer {
            n_out,
            cell: LSTMCell::new(n_out, act_fn, gate_fn, init, optimizer),
        }
    }

    /// Forward pass over a full sequence.
    pub fn forward(&mut self, x: &Array2<f64>, n_t: usize) -> Vec<Array2<f64>> {
        let mut outputs = Vec::with_capacity(n_t);
        let n_ex = x.nrows() / n_t;
        for t in 0..n_t {
            let xt = x.slice(s![t * n_ex..(t + 1) * n_ex, ..]).to_owned();
            let (at, _ct) = self.cell.forward(&xt);
            outputs.push(at);
        }
        outputs
    }

    /// Backward pass over a full sequence.
    pub fn backward(&mut self, dldy: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let mut dxs = Vec::new();
        for t in (0..dldy.len()).rev() {
            let (dxt, _da_prev) = self.cell.backward(&dldy[t]);
            dxs.insert(0, dxt);
        }
        dxs
    }
}

/// Dropout layer.
pub struct Dropout {
    pub p: f64,
    pub is_training: bool,
    mask: Option<Array2<f64>>,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            is_training: true,
            mask: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>, retain_derived: bool) -> Array2<f64> {
        if !self.is_training || self.p == 0.0 {
            return x.clone();
        }
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);
        let mask = Array2::from_shape_fn(x.raw_dim(), |_| {
            if rng.r#gen::<f64>() >= self.p {
                scale
            } else {
                0.0
            }
        });
        let y = x * &mask;
        if retain_derived {
            self.mask = Some(mask);
        }
        y
    }

    pub fn backward(&mut self, dldy: &Array2<f64>) -> Array2<f64> {
        if !self.is_training || self.p == 0.0 {
            return dldy.clone();
        }
        let mask = self.mask.take().expect("No cached mask for backward");
        dldy * &mask
    }
}

/// Batch Normalization for 2D inputs (batch_size, n_features).
pub struct BatchNorm {
    pub n_in: usize,
    pub momentum: f64,
    pub eps: f64,
    pub is_training: bool,
    pub gamma: Array1<f64>,
    pub beta: Array1<f64>,
    pub running_mean: Array1<f64>,
    pub running_var: Array1<f64>,
    pub dgamma: Option<Array1<f64>>,
    pub dbeta: Option<Array1<f64>>,
    x_hat_cache: Option<Array2<f64>>,
    std_inv_cache: Option<Array1<f64>>,
}

impl BatchNorm {
    pub fn new(n_in: usize, momentum: Option<f64>, eps: Option<f64>) -> Self {
        Self {
            n_in,
            momentum: momentum.unwrap_or(0.9),
            eps: eps.unwrap_or(1e-5),
            is_training: true,
            gamma: Array1::ones(n_in),
            beta: Array1::zeros(n_in),
            running_mean: Array1::zeros(n_in),
            running_var: Array1::ones(n_in),
            dgamma: None,
            dbeta: None,
            x_hat_cache: None,
            std_inv_cache: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>, retain_derived: bool) -> Array2<f64> {
        let (batch_size, n_features) = x.dim();
        assert_eq!(n_features, self.n_in);

        if self.is_training {
            let mean = x.mean_axis(ndarray::Axis(0)).unwrap();
            let var = x.var_axis(ndarray::Axis(0), 0.0);

            self.running_mean = &self.running_mean * self.momentum + &mean * (1.0 - self.momentum);
            self.running_var = &self.running_var * self.momentum + &var * (1.0 - self.momentum);

            let std_inv = var.mapv(|v| 1.0 / (v + self.eps).sqrt());
            let mut x_hat = Array2::zeros((batch_size, n_features));
            let mut y = Array2::zeros((batch_size, n_features));

            for r in 0..batch_size {
                for c in 0..n_features {
                    let norm = (x[[r, c]] - mean[c]) * std_inv[c];
                    x_hat[[r, c]] = norm;
                    y[[r, c]] = norm * self.gamma[c] + self.beta[c];
                }
            }

            if retain_derived {
                self.x_hat_cache = Some(x_hat);
                self.std_inv_cache = Some(std_inv);
            }

            y
        } else {
            let std_inv = self.running_var.mapv(|v| 1.0 / (v + self.eps).sqrt());
            let mut y = Array2::zeros((batch_size, n_features));
            for r in 0..batch_size {
                for c in 0..n_features {
                    let norm = (x[[r, c]] - self.running_mean[c]) * std_inv[c];
                    y[[r, c]] = norm * self.gamma[c] + self.beta[c];
                }
            }
            y
        }
    }

    pub fn backward(&mut self, dldy: &Array2<f64>) -> Array2<f64> {
        let (batch_size, n_features) = dldy.dim();
        let m = batch_size as f64;
        let x_hat = self.x_hat_cache.take().expect("No cached x_hat");
        let std_inv = self.std_inv_cache.take().expect("No cached std_inv");

        let dbeta = dldy.sum_axis(ndarray::Axis(0));
        let dgamma = (dldy * &x_hat).sum_axis(ndarray::Axis(0));

        let mut dx = Array2::zeros((batch_size, n_features));
        for c in 0..n_features {
            let g = self.gamma[c];
            let inv = std_inv[c];
            let sum_dy = dbeta[c];
            let sum_dy_xhat = dgamma[c];

            for r in 0..batch_size {
                let dy = dldy[[r, c]];
                let xh = x_hat[[r, c]];
                dx[[r, c]] = (g * inv / m) * (m * dy - sum_dy - xh * sum_dy_xhat);
            }
        }

        self.dbeta = Some(dbeta);
        self.dgamma = Some(dgamma);
        dx
    }
}

/// Embedding layer that maps integer tokens to dense continuous vectors.
pub struct Embedding {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub weights: Array2<f64>,
    pub dweights: Option<Array2<f64>>,
    cache_indices: Option<Vec<usize>>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let gain = 1.0;
        let weights = initializers::glorot_uniform(vocab_size, embed_dim, gain);
        Self {
            vocab_size,
            embed_dim,
            weights,
            dweights: None,
            cache_indices: None,
        }
    }

    pub fn forward(&mut self, indices: &[usize], retain_derived: bool) -> Array2<f64> {
        let n = indices.len();
        let mut out = Array2::zeros((n, self.embed_dim));
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.vocab_size {
                for d in 0..self.embed_dim {
                    out[[i, d]] = self.weights[[idx, d]];
                }
            }
        }
        if retain_derived {
            self.cache_indices = Some(indices.to_vec());
        }
        out
    }

    pub fn backward(&mut self, dldy: &Array2<f64>) {
        let indices = self.cache_indices.take().expect("No cached indices");
        let mut dweights = Array2::zeros(self.weights.raw_dim());
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.vocab_size {
                for d in 0..self.embed_dim {
                    dweights[[idx, d]] += dldy[[i, d]];
                }
            }
        }
        self.dweights = Some(dweights);
    }
}

/// Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
pub struct Attention {
    pub scale: f64,
    cache_q: Option<Array2<f64>>,
    cache_k: Option<Array2<f64>>,
    cache_v: Option<Array2<f64>>,
    cache_weights: Option<Array2<f64>>,
}

impl Attention {
    pub fn new() -> Self {
        Self {
            scale: 1.0,
            cache_q: None,
            cache_k: None,
            cache_v: None,
            cache_weights: None,
        }
    }

    pub fn forward(
        &mut self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        mask: Option<&Array2<f64>>,
        retain_derived: bool,
    ) -> Array2<f64> {
        let dk = q.ncols() as f64;
        let scale = 1.0 / dk.sqrt();
        self.scale = scale;

        let mut scores = q.dot(&k.t()) * scale;
        if let Some(m) = mask {
            scores += m;
        }

        let weights = super::utils::softmax(&scores);
        let out = weights.dot(v);

        if retain_derived {
            self.cache_q = Some(q.clone());
            self.cache_k = Some(k.clone());
            self.cache_v = Some(v.clone());
            self.cache_weights = Some(weights);
        }

        out
    }

    pub fn backward(&mut self, dldout: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let q = self.cache_q.take().unwrap();
        let k = self.cache_k.take().unwrap();
        let v = self.cache_v.take().unwrap();
        let weights = self.cache_weights.take().unwrap();

        let dv = weights.t().dot(dldout);
        let dweights = dldout.dot(&v.t());

        let (n_rows, n_cols) = weights.dim();
        let mut dscores = Array2::zeros((n_rows, n_cols));
        for i in 0..n_rows {
            let row_dot: f64 = (0..n_cols).map(|j| dweights[[i, j]] * weights[[i, j]]).sum();
            for j in 0..n_cols {
                dscores[[i, j]] = weights[[i, j]] * (dweights[[i, j]] - row_dot) * self.scale;
            }
        }

        let dq = dscores.dot(&k);
        let dk = dscores.t().dot(&q);

        (dq, dk, dv)
    }
}

impl Default for Attention {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-head Attention layer.
pub struct MultiHeadAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_k: usize,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    attn: Attention,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_k = d_model / n_heads;
        Self {
            d_model,
            n_heads,
            d_k,
            w_q: Linear::new(d_model, Some("Identity"), "glorot_uniform", Some("SGD")),
            w_k: Linear::new(d_model, Some("Identity"), "glorot_uniform", Some("SGD")),
            w_v: Linear::new(d_model, Some("Identity"), "glorot_uniform", Some("SGD")),
            w_o: Linear::new(d_model, Some("Identity"), "glorot_uniform", Some("SGD")),
            attn: Attention::new(),
        }
    }

    pub fn forward(
        &mut self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        mask: Option<&Array2<f64>>,
        retain_derived: bool,
    ) -> Array2<f64> {
        let q_proj = self.w_q.forward(q, retain_derived);
        let k_proj = self.w_k.forward(k, retain_derived);
        let v_proj = self.w_v.forward(v, retain_derived);

        let head_out = self.attn.forward(&q_proj, &k_proj, &v_proj, mask, retain_derived);
        self.w_o.forward(&head_out, retain_derived)
    }

    pub fn backward(&mut self, dldout: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let d_head = self.w_o.backward(dldout, true);
        let (dq_proj, dk_proj, dv_proj) = self.attn.backward(&d_head);
        let dq = self.w_q.backward(&dq_proj, true);
        let dk = self.w_k.backward(&dk_proj, true);
        let dv = self.w_v.backward(&dv_proj, true);
        (dq, dk, dv)
    }
}

/// Gated Recurrent Unit (GRU) cell.
pub struct GRUCell {
    pub n_in: usize,
    pub n_out: usize,
    pub trainable: bool,
    pub wxr: Array2<f64>,
    pub whr: Array2<f64>,
    pub br: Array2<f64>,
    pub wxz: Array2<f64>,
    pub whz: Array2<f64>,
    pub bz: Array2<f64>,
    pub wxh: Array2<f64>,
    pub whh: Array2<f64>,
    pub bh: Array2<f64>,
    pub dwxr: Array2<f64>,
    pub dwhr: Array2<f64>,
    pub dbr: Array2<f64>,
    pub dwxz: Array2<f64>,
    pub dwhz: Array2<f64>,
    pub dbz: Array2<f64>,
    pub dwxh: Array2<f64>,
    pub dwhh: Array2<f64>,
    pub dbh: Array2<f64>,
    h_cache: Vec<Array2<f64>>,
    x_cache: Vec<Array2<f64>>,
    r_cache: Vec<Array2<f64>>,
    z_cache: Vec<Array2<f64>>,
    n_cache: Vec<Array2<f64>>,
    current_step: usize,
    dldh_acc: Option<Array2<f64>>,
}

impl GRUCell {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let gain = 1.0;
        let wxr = initializers::glorot_uniform(n_in, n_out, gain);
        let whr = initializers::glorot_uniform(n_out, n_out, gain);
        let br = Array2::zeros((1, n_out));

        let wxz = initializers::glorot_uniform(n_in, n_out, gain);
        let whz = initializers::glorot_uniform(n_out, n_out, gain);
        let bz = Array2::zeros((1, n_out));

        let wxh = initializers::glorot_uniform(n_in, n_out, gain);
        let whh = initializers::glorot_uniform(n_out, n_out, gain);
        let bh = Array2::zeros((1, n_out));

        Self {
            n_in,
            n_out,
            trainable: true,
            wxr: wxr.clone(),
            whr: whr.clone(),
            br: br.clone(),
            wxz: wxz.clone(),
            whz: whz.clone(),
            bz: bz.clone(),
            wxh: wxh.clone(),
            whh: whh.clone(),
            bh: bh.clone(),
            dwxr: Array2::zeros(wxr.raw_dim()),
            dwhr: Array2::zeros(whr.raw_dim()),
            dbr: Array2::zeros(br.raw_dim()),
            dwxz: Array2::zeros(wxz.raw_dim()),
            dwhz: Array2::zeros(whz.raw_dim()),
            dbz: Array2::zeros(bz.raw_dim()),
            dwxh: Array2::zeros(wxh.raw_dim()),
            dwhh: Array2::zeros(whh.raw_dim()),
            dbh: Array2::zeros(bh.raw_dim()),
            h_cache: Vec::new(),
            x_cache: Vec::new(),
            r_cache: Vec::new(),
            z_cache: Vec::new(),
            n_cache: Vec::new(),
            current_step: 0,
            dldh_acc: None,
        }
    }

    pub fn forward(&mut self, xt: &Array2<f64>) -> Array2<f64> {
        let h_prev = if self.h_cache.is_empty() {
            Array2::zeros((xt.nrows(), self.n_out))
        } else {
            self.h_cache.last().unwrap().clone()
        };

        let r = (xt.dot(&self.wxr) + h_prev.dot(&self.whr) + &self.br)
            .mapv(|v| 1.0 / (1.0 + (-v).exp()));

        let z = (xt.dot(&self.wxz) + h_prev.dot(&self.whz) + &self.bz)
            .mapv(|v| 1.0 / (1.0 + (-v).exp()));

        let rh = &r * &h_prev;
        let n = (xt.dot(&self.wxh) + rh.dot(&self.whh) + &self.bh).mapv(|v| v.tanh());

        let h = (1.0 - &z) * &n + &z * &h_prev;

        self.x_cache.push(xt.clone());
        self.r_cache.push(r);
        self.z_cache.push(z);
        self.n_cache.push(n);
        self.h_cache.push(h.clone());

        h
    }

    pub fn backward(&mut self, dldht: &Array2<f64>) -> Array2<f64> {
        if self.current_step == 0 {
            self.current_step = self.h_cache.len() - 1;
        }
        let t = self.current_step;
        let h_prev = if t > 0 {
            &self.h_cache[t - 1]
        } else {
            &self.h_cache[0]
        };
        let xt = &self.x_cache[t];
        let r = &self.r_cache[t];
        let z = &self.z_cache[t];
        let n = &self.n_cache[t];

        let acc = self.dldh_acc.take().unwrap_or_else(|| Array2::zeros(dldht.raw_dim()));
        let dh = dldht + &acc;

        let dn = &dh * (1.0 - z);
        let dz = &dh * (h_prev - n);

        let dn_raw = dn * n.mapv(|v| 1.0 - v * v);
        let dz_raw = dz * z.mapv(|v| v * (1.0 - v));

        let drh = dn_raw.dot(&self.whh.t());
        let dr = &drh * h_prev;
        let dr_raw = dr * r.mapv(|v| v * (1.0 - v));

        self.dwxh += &xt.t().dot(&dn_raw);
        self.dwhh += &(r * h_prev).t().dot(&dn_raw);
        self.dbh += &dn_raw.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));

        self.dwxz += &xt.t().dot(&dz_raw);
        self.dwhz += &h_prev.t().dot(&dz_raw);
        self.dbz += &dz_raw.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));

        self.dwxr += &xt.t().dot(&dr_raw);
        self.dwhr += &h_prev.t().dot(&dr_raw);
        self.dbr += &dr_raw.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));

        let dx = dn_raw.dot(&self.wxh.t()) + dz_raw.dot(&self.wxz.t()) + dr_raw.dot(&self.wxr.t());
        let dh_prev = &dh * z + drh * r + dz_raw.dot(&self.whz.t()) + dr_raw.dot(&self.whr.t());
        self.dldh_acc = Some(dh_prev);
        self.current_step = t.wrapping_sub(1);

        dx
    }
}

/// Full sequence GRU layer.
pub struct GRULayer {
    pub n_out: usize,
    pub cell: GRUCell,
}

impl GRULayer {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        Self {
            n_out,
            cell: GRUCell::new(n_in, n_out),
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>, n_t: usize) -> Vec<Array2<f64>> {
        let mut outputs = Vec::with_capacity(n_t);
        let n_ex = x.nrows() / n_t;
        for t in 0..n_t {
            let xt = x.slice(s![t * n_ex..(t + 1) * n_ex, ..]).to_owned();
            let ht = self.cell.forward(&xt);
            outputs.push(ht);
        }
        outputs
    }

    pub fn backward(&mut self, dldy: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let mut dxs = Vec::new();
        for t in (0..dldy.len()).rev() {
            let dxt = self.cell.backward(&dldy[t]);
            dxs.insert(0, dxt);
        }
        dxs
    }
}

/// 1D Convolutional Layer.
pub struct Conv1D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub pad: usize,
    pub weights: Array3<f64>,
    pub bias: Array1<f64>,
    pub dweights: Option<Array3<f64>>,
    pub dbias: Option<Array1<f64>>,
    x_cache: Option<Array3<f64>>,
}

impl Conv1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        pad: Option<usize>,
    ) -> Self {
        let stride = stride.unwrap_or(1);
        let pad = pad.unwrap_or(0);
        let gain = 1.0;
        let mut rng = rand::thread_rng();
        let bound = (gain / (in_channels * kernel_size) as f64).sqrt();

        let weights = Array3::from_shape_fn((kernel_size, in_channels, out_channels), |_| {
            rng.gen_range(-bound..bound)
        });
        let bias = Array1::zeros(out_channels);

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad,
            weights,
            bias,
            dweights: None,
            dbias: None,
            x_cache: None,
        }
    }

    pub fn forward(&mut self, x: &Array3<f64>, retain_derived: bool) -> Array3<f64> {
        let (batch_size, length, in_c) = x.dim();
        assert_eq!(in_c, self.in_channels);

        let k = self.kernel_size;
        let s = self.stride;
        let p = self.pad;
        let out_len = (length + 2 * p - k) / s + 1;

        let mut x_pad = Array3::<f64>::zeros((batch_size, length + 2 * p, in_c));
        for b in 0..batch_size {
            for l in 0..length {
                for c in 0..in_c {
                    x_pad[[b, l + p, c]] = x[[b, l, c]];
                }
            }
        }

        let mut out = Array3::<f64>::zeros((batch_size, out_len, self.out_channels));
        for b in 0..batch_size {
            for ol in 0..out_len {
                let start = ol * s;
                for oc in 0..self.out_channels {
                    let mut sum = self.bias[oc];
                    for ki in 0..k {
                        for ic in 0..in_c {
                            sum += x_pad[[b, start + ki, ic]] * self.weights[[ki, ic, oc]];
                        }
                    }
                    out[[b, ol, oc]] = sum;
                }
            }
        }

        if retain_derived {
            self.x_cache = Some(x_pad);
        }
        out
    }

    pub fn backward(&mut self, dldy: &Array3<f64>) -> Array3<f64> {
        let x_pad = self.x_cache.take().expect("No cached input for Conv1D backward");
        let (batch_size, padded_len, in_c) = x_pad.dim();
        let (_b, out_len, out_c) = dldy.dim();
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.pad;
        let orig_len = padded_len - 2 * p;

        let mut dweights = Array3::<f64>::zeros(self.weights.raw_dim());
        let mut dbias = Array1::<f64>::zeros(self.out_channels);
        let mut dx_pad = Array3::<f64>::zeros((batch_size, padded_len, in_c));

        for b in 0..batch_size {
            for ol in 0..out_len {
                let start = ol * s;
                for oc in 0..out_c {
                    let dy = dldy[[b, ol, oc]];
                    dbias[oc] += dy;
                    for ki in 0..k {
                        for ic in 0..in_c {
                            dweights[[ki, ic, oc]] += dy * x_pad[[b, start + ki, ic]];
                            dx_pad[[b, start + ki, ic]] += dy * self.weights[[ki, ic, oc]];
                        }
                    }
                }
            }
        }

        self.dweights = Some(dweights);
        self.dbias = Some(dbias);

        let mut dx = Array3::<f64>::zeros((batch_size, orig_len, in_c));
        for b in 0..batch_size {
            for l in 0..orig_len {
                for c in 0..in_c {
                    dx[[b, l, c]] = dx_pad[[b, l + p, c]];
                }
            }
        }
        dx
    }
}

/// 2D Convolutional Layer.
pub struct Conv2D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub pad: usize,
    pub weights: Array4<f64>,
    pub bias: Array1<f64>,
    pub dweights: Option<Array4<f64>>,
    pub dbias: Option<Array1<f64>>,
    x_cache: Option<Array4<f64>>,
}

impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_shape: (usize, usize),
        stride: Option<usize>,
        pad: Option<usize>,
    ) -> Self {
        let (kh, kw) = kernel_shape;
        let stride = stride.unwrap_or(1);
        let pad = pad.unwrap_or(0);
        let gain = 1.0;
        let bound = (gain / (in_channels * kh * kw) as f64).sqrt();
        let mut rng = rand::thread_rng();

        let weights = Array4::from_shape_fn((kh, kw, in_channels, out_channels), |_| {
            rng.gen_range(-bound..bound)
        });
        let bias = Array1::zeros(out_channels);

        Self {
            in_channels,
            out_channels,
            kernel_h: kh,
            kernel_w: kw,
            stride,
            pad,
            weights,
            bias,
            dweights: None,
            dbias: None,
            x_cache: None,
        }
    }

    pub fn forward(&mut self, x: &Array4<f64>, retain_derived: bool) -> Array4<f64> {
        let (bs, h, w, in_c) = x.dim();
        assert_eq!(in_c, self.in_channels);
        let p = self.pad;
        let s = self.stride;
        let out_h = (h + 2 * p - self.kernel_h) / s + 1;
        let out_w = (w + 2 * p - self.kernel_w) / s + 1;

        let mut x_pad = Array4::<f64>::zeros((bs, h + 2 * p, w + 2 * p, in_c));
        for b in 0..bs {
            for i in 0..h {
                for j in 0..w {
                    for c in 0..in_c {
                        x_pad[[b, i + p, j + p, c]] = x[[b, i, j, c]];
                    }
                }
            }
        }

        let mut out = Array4::<f64>::zeros((bs, out_h, out_w, self.out_channels));
        for b in 0..bs {
            for oh in 0..out_h {
                let start_h = oh * s;
                for ow in 0..out_w {
                    let start_w = ow * s;
                    for oc in 0..self.out_channels {
                        let mut sum = self.bias[oc];
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                for ic in 0..in_c {
                                    sum += x_pad[[b, start_h + kh, start_w + kw, ic]]
                                        * self.weights[[kh, kw, ic, oc]];
                                }
                            }
                        }
                        out[[b, oh, ow, oc]] = sum;
                    }
                }
            }
        }

        if retain_derived {
            self.x_cache = Some(x_pad);
        }
        out
    }

    pub fn backward(&mut self, dldy: &Array4<f64>) -> Array4<f64> {
        let x_pad = self.x_cache.take().expect("No cached input for Conv2D backward");
        let (bs, pad_h, pad_w, in_c) = x_pad.dim();
        let (_b, out_h, out_w, out_c) = dldy.dim();
        let s = self.stride;
        let p = self.pad;
        let orig_h = pad_h - 2 * p;
        let orig_w = pad_w - 2 * p;

        let mut dweights = Array4::<f64>::zeros(self.weights.raw_dim());
        let mut dbias = Array1::<f64>::zeros(self.out_channels);
        let mut dx_pad = Array4::<f64>::zeros((bs, pad_h, pad_w, in_c));

        for b in 0..bs {
            for oh in 0..out_h {
                let start_h = oh * s;
                for ow in 0..out_w {
                    let start_w = ow * s;
                    for oc in 0..out_c {
                        let dy = dldy[[b, oh, ow, oc]];
                        dbias[oc] += dy;
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                for ic in 0..in_c {
                                    dweights[[kh, kw, ic, oc]] +=
                                        dy * x_pad[[b, start_h + kh, start_w + kw, ic]];
                                    dx_pad[[b, start_h + kh, start_w + kw, ic]] +=
                                        dy * self.weights[[kh, kw, ic, oc]];
                                }
                            }
                        }
                    }
                }
            }
        }

        self.dweights = Some(dweights);
        self.dbias = Some(dbias);

        let mut dx = Array4::<f64>::zeros((bs, orig_h, orig_w, in_c));
        for b in 0..bs {
            for i in 0..orig_h {
                for j in 0..orig_w {
                    for c in 0..in_c {
                        dx[[b, i, j, c]] = dx_pad[[b, i + p, j + p, c]];
                    }
                }
            }
        }
        dx
    }
}

/// 1D Max Pooling Layer.
pub struct MaxPool1D {
    pub kernel_size: usize,
    pub stride: usize,
    argmax_cache: Option<Array3<usize>>,
    orig_len: usize,
}

impl MaxPool1D {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            argmax_cache: None,
            orig_len: 0,
        }
    }

    pub fn forward(&mut self, x: &Array3<f64>, retain_derived: bool) -> Array3<f64> {
        let (bs, length, channels) = x.dim();
        self.orig_len = length;
        let k = self.kernel_size;
        let s = self.stride;
        let out_len = (length - k) / s + 1;

        let mut out = Array3::<f64>::zeros((bs, out_len, channels));
        let mut argmax = Array3::<usize>::zeros((bs, out_len, channels));

        for b in 0..bs {
            for ol in 0..out_len {
                let start = ol * s;
                for c in 0..channels {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = 0;
                    for ki in 0..k {
                        let val = x[[b, start + ki, c]];
                        if val > max_val {
                            max_val = val;
                            max_idx = start + ki;
                        }
                    }
                    out[[b, ol, c]] = max_val;
                    argmax[[b, ol, c]] = max_idx;
                }
            }
        }

        if retain_derived {
            self.argmax_cache = Some(argmax);
        }
        out
    }

    pub fn backward(&mut self, dldy: &Array3<f64>) -> Array3<f64> {
        let argmax = self.argmax_cache.take().expect("No cached argmax for MaxPool1D");
        let (bs, out_len, channels) = dldy.dim();
        let mut dx = Array3::<f64>::zeros((bs, self.orig_len, channels));

        for b in 0..bs {
            for ol in 0..out_len {
                for c in 0..channels {
                    let idx = argmax[[b, ol, c]];
                    dx[[b, idx, c]] += dldy[[b, ol, c]];
                }
            }
        }
        dx
    }
}

/// 1D Average Pooling Layer.
pub struct AvgPool1D {
    pub kernel_size: usize,
    pub stride: usize,
    orig_len: usize,
}

impl AvgPool1D {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            orig_len: 0,
        }
    }

    pub fn forward(&mut self, x: &Array3<f64>) -> Array3<f64> {
        let (bs, length, channels) = x.dim();
        self.orig_len = length;
        let k = self.kernel_size;
        let s = self.stride;
        let out_len = (length - k) / s + 1;

        let mut out = Array3::<f64>::zeros((bs, out_len, channels));
        for b in 0..bs {
            for ol in 0..out_len {
                let start = ol * s;
                for c in 0..channels {
                    let sum: f64 = (0..k).map(|ki| x[[b, start + ki, c]]).sum();
                    out[[b, ol, c]] = sum / k as f64;
                }
            }
        }
        out
    }

    pub fn backward(&self, dldy: &Array3<f64>) -> Array3<f64> {
        let (bs, out_len, channels) = dldy.dim();
        let k = self.kernel_size;
        let s = self.stride;
        let mut dx = Array3::<f64>::zeros((bs, self.orig_len, channels));

        for b in 0..bs {
            for ol in 0..out_len {
                let start = ol * s;
                for c in 0..channels {
                    let grad = dldy[[b, ol, c]] / k as f64;
                    for ki in 0..k {
                        dx[[b, start + ki, c]] += grad;
                    }
                }
            }
        }
        dx
    }
}

/// 2D Max Pooling Layer.
pub struct MaxPool2D {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    argmax_cache: Option<Array4<(usize, usize)>>,
    orig_h: usize,
    orig_w: usize,
}

impl MaxPool2D {
    pub fn new(kernel_shape: (usize, usize), stride: Option<usize>) -> Self {
        Self {
            kernel_h: kernel_shape.0,
            kernel_w: kernel_shape.1,
            stride: stride.unwrap_or(kernel_shape.0),
            argmax_cache: None,
            orig_h: 0,
            orig_w: 0,
        }
    }

    pub fn forward(&mut self, x: &Array4<f64>, retain_derived: bool) -> Array4<f64> {
        let (bs, h, w, c) = x.dim();
        self.orig_h = h;
        self.orig_w = w;
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride;
        let out_h = (h - kh) / s + 1;
        let out_w = (w - kw) / s + 1;

        let mut out = Array4::<f64>::zeros((bs, out_h, out_w, c));
        let mut argmax = Array4::from_elem((bs, out_h, out_w, c), (0usize, 0usize));

        for b in 0..bs {
            for oh in 0..out_h {
                let start_h = oh * s;
                for ow in 0..out_w {
                    let start_w = ow * s;
                    for ch in 0..c {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_pos = (0, 0);
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let val = x[[b, start_h + ki, start_w + kj, ch]];
                                if val > max_val {
                                    max_val = val;
                                    max_pos = (start_h + ki, start_w + kj);
                                }
                            }
                        }
                        out[[b, oh, ow, ch]] = max_val;
                        argmax[[b, oh, ow, ch]] = max_pos;
                    }
                }
            }
        }

        if retain_derived {
            self.argmax_cache = Some(argmax);
        }
        out
    }

    pub fn backward(&mut self, dldy: &Array4<f64>) -> Array4<f64> {
        let argmax = self.argmax_cache.take().expect("No cached argmax for MaxPool2D");
        let (bs, out_h, out_w, c) = dldy.dim();
        let mut dx = Array4::<f64>::zeros((bs, self.orig_h, self.orig_w, c));

        for b in 0..bs {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ch in 0..c {
                        let (pos_h, pos_w) = argmax[[b, oh, ow, ch]];
                        dx[[b, pos_h, pos_w, ch]] += dldy[[b, oh, ow, ch]];
                    }
                }
            }
        }
        dx
    }
}

/// 2D Average Pooling Layer.
pub struct AvgPool2D {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    orig_h: usize,
    orig_w: usize,
}

impl AvgPool2D {
    pub fn new(kernel_shape: (usize, usize), stride: Option<usize>) -> Self {
        Self {
            kernel_h: kernel_shape.0,
            kernel_w: kernel_shape.1,
            stride: stride.unwrap_or(kernel_shape.0),
            orig_h: 0,
            orig_w: 0,
        }
    }

    pub fn forward(&mut self, x: &Array4<f64>) -> Array4<f64> {
        let (bs, h, w, c) = x.dim();
        self.orig_h = h;
        self.orig_w = w;
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride;
        let out_h = (h - kh) / s + 1;
        let out_w = (w - kw) / s + 1;
        let area = (kh * kw) as f64;

        let mut out = Array4::<f64>::zeros((bs, out_h, out_w, c));
        for b in 0..bs {
            for oh in 0..out_h {
                let start_h = oh * s;
                for ow in 0..out_w {
                    let start_w = ow * s;
                    for ch in 0..c {
                        let mut sum = 0.0;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                sum += x[[b, start_h + ki, start_w + kj, ch]];
                            }
                        }
                        out[[b, oh, ow, ch]] = sum / area;
                    }
                }
            }
        }
        out
    }

    pub fn backward(&self, dldy: &Array4<f64>) -> Array4<f64> {
        let (bs, out_h, out_w, c) = dldy.dim();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let s = self.stride;
        let area = (kh * kw) as f64;
        let mut dx = Array4::<f64>::zeros((bs, self.orig_h, self.orig_w, c));

        for b in 0..bs {
            for oh in 0..out_h {
                let start_h = oh * s;
                for ow in 0..out_w {
                    let start_w = ow * s;
                    for ch in 0..c {
                        let grad = dldy[[b, oh, ow, ch]] / area;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                dx[[b, start_h + ki, start_w + kj, ch]] += grad;
                            }
                        }
                    }
                }
            }
        }
        dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_forward() {
        let mut layer = Linear::new(3, Some("ReLU"), "glorot_uniform", Some("SGD"));
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = layer.forward(&x, false);
        assert_eq!(y.dim(), (2, 3));
    }

    #[test]
    fn test_linear_backward() {
        let mut layer = Linear::new(3, Some("ReLU"), "glorot_uniform", Some("SGD"));
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = layer.forward(&x, true);
        let dldy = Array2::ones(y.raw_dim());
        let dx = layer.backward(&dldy, true);
        assert_eq!(dx.dim(), x.dim());
    }

    #[test]
    fn test_rnn_cell_forward() {
        let mut cell = RNNCell::new(4, Some("Tanh"), "glorot_uniform", Some("SGD"));
        let xt = array![[1.0, 2.0, 3.0]];
        let at = cell.forward(&xt);
        assert_eq!(at.dim(), (1, 4));
    }

    #[test]
    fn test_lstm_cell_forward() {
        let mut cell = LSTMCell::new(4, Some("Tanh"), Some("Sigmoid"), "glorot_uniform", Some("SGD"));
        let xt = array![[1.0, 2.0, 3.0]];
        let (at, ct) = cell.forward(&xt);
        assert_eq!(at.dim(), (1, 4));
        assert_eq!(ct.dim(), (1, 4));
    }

    #[test]
    fn test_dropout() {
        let mut drop = Dropout::new(0.5);
        drop.is_training = false;
        let x = Array2::from_elem((10, 10), 2.0);
        let out = drop.forward(&x, false);
        assert_eq!(out, x);

        drop.is_training = true;
        let out_train = drop.forward(&x, true);
        assert_eq!(out_train.dim(), x.dim());
        let dldout = Array2::ones((10, 10));
        let dx = drop.backward(&dldout);
        assert_eq!(dx.dim(), (10, 10));
    }

    #[test]
    fn test_batch_norm() {
        let mut bn = BatchNorm::new(2, Some(0.9), Some(1e-5));
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = bn.forward(&x, true);
        assert_eq!(y.dim(), (3, 2));

        let dldy = Array2::ones((3, 2));
        let dx = bn.backward(&dldy);
        assert_eq!(dx.dim(), (3, 2));

        // Inference forward
        let y_inf = bn.forward(&x, false);
        assert_eq!(y_inf.dim(), (3, 2));
    }

    #[test]
    fn test_embedding() {
        let mut emb = Embedding::new(10, 8);
        let indices = vec![1, 4, 2];
        let out = emb.forward(&indices, true);
        assert_eq!(out.dim(), (3, 8));

        let dldout = Array2::ones((3, 8));
        let () = emb.backward(&dldout);
        assert!(emb.dweights.is_some());
    }

    #[test]
    fn test_attention() {
        let mut attn = Attention::new();
        let q = array![[1.0, 0.0], [0.0, 1.0]];
        let k = array![[1.0, 0.0], [0.0, 1.0]];
        let v = array![[2.0, 3.0], [4.0, 5.0]];
        let out = attn.forward(&q, &k, &v, None, true);
        assert_eq!(out.dim(), (2, 2));

        let dldout = Array2::ones((2, 2));
        let (dq, dk, dv) = attn.backward(&dldout);
        assert_eq!(dq.dim(), (2, 2));
        assert_eq!(dk.dim(), (2, 2));
        assert_eq!(dv.dim(), (2, 2));
    }

    #[test]
    fn test_multihead_attention() {
        let mut mha = MultiHeadAttention::new(8, 2);
        let q = Array2::ones((4, 8));
        let k = Array2::ones((4, 8));
        let v = Array2::ones((4, 8));
        let out = mha.forward(&q, &k, &v, None, true);
        assert_eq!(out.dim(), (4, 8));

        let dldout = Array2::ones((4, 8));
        let (dq, dk, dv) = mha.backward(&dldout);
        assert_eq!(dq.dim(), (4, 8));
        assert_eq!(dk.dim(), (4, 8));
        assert_eq!(dv.dim(), (4, 8));
    }

    #[test]
    fn test_gru_cell_and_layer() {
        let mut gru = GRUCell::new(3, 4);
        let xt = array![[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]];
        let ht = gru.forward(&xt);
        assert_eq!(ht.dim(), (2, 4));

        let dldh = Array2::ones((2, 4));
        let dxt = gru.backward(&dldh);
        assert_eq!(dxt.dim(), (2, 3));

        let mut gru_layer = GRULayer::new(3, 4);
        // 2 timesteps, batch size 2 -> 4 rows
        let x_seq = Array2::ones((4, 3));
        let h_seq = gru_layer.forward(&x_seq, 2);
        assert_eq!(h_seq.len(), 2);
        assert_eq!(h_seq[0].dim(), (2, 4));

        let grads = vec![Array2::ones((2, 4)), Array2::ones((2, 4))];
        let dx_seq = gru_layer.backward(&grads);
        assert_eq!(dx_seq.len(), 2);
        assert_eq!(dx_seq[0].dim(), (2, 3));
    }

    #[test]
    fn test_conv1d_and_pooling1d() {
        let mut conv = Conv1D::new(2, 3, 3, Some(1), Some(1));
        let x = Array3::ones((2, 5, 2));
        let out = conv.forward(&x, true);
        assert_eq!(out.dim(), (2, 5, 3));

        let dldout = Array3::ones((2, 5, 3));
        let dx = conv.backward(&dldout);
        assert_eq!(dx.dim(), (2, 5, 2));

        let mut maxpool = MaxPool1D::new(2, Some(2));
        let mp_out = maxpool.forward(&x, true);
        assert_eq!(mp_out.dim(), (2, 2, 2));
        let dmp = maxpool.backward(&Array3::ones((2, 2, 2)));
        assert_eq!(dmp.dim(), (2, 5, 2));

        let mut avgpool = AvgPool1D::new(2, Some(2));
        let ap_out = avgpool.forward(&x);
        assert_eq!(ap_out.dim(), (2, 2, 2));
        let dap = avgpool.backward(&Array3::ones((2, 2, 2)));
        assert_eq!(dap.dim(), (2, 5, 2));
    }

    #[test]
    fn test_conv2d_and_pooling2d() {
        let mut conv = Conv2D::new(1, 2, (3, 3), Some(1), Some(1));
        let x = Array4::ones((2, 4, 4, 1));
        let out = conv.forward(&x, true);
        assert_eq!(out.dim(), (2, 4, 4, 2));

        let dldout = Array4::ones((2, 4, 4, 2));
        let dx = conv.backward(&dldout);
        assert_eq!(dx.dim(), (2, 4, 4, 1));

        let mut maxpool = MaxPool2D::new((2, 2), Some(2));
        let mp_out = maxpool.forward(&x, true);
        assert_eq!(mp_out.dim(), (2, 2, 2, 1));
        let dmp = maxpool.backward(&Array4::ones((2, 2, 2, 1)));
        assert_eq!(dmp.dim(), (2, 4, 4, 1));

        let mut avgpool = AvgPool2D::new((2, 2), Some(2));
        let ap_out = avgpool.forward(&x);
        assert_eq!(ap_out.dim(), (2, 2, 2, 1));
        let dap = avgpool.backward(&Array4::ones((2, 2, 2, 1)));
        assert_eq!(dap.dim(), (2, 4, 4, 1));
    }
}

