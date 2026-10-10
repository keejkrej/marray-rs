# marray

[![Crates.io](https://img.shields.io/crates/v/marray.svg)](https://crates.io/crates/marray)
[![Documentation](https://docs.rs/marray/badge.svg)](https://docs.rs/marray)
[![CI](https://github.com/keejkrej/marray-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/keejkrej/marray-rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

A comprehensive, high-performance, pure-Rust scientific computing and machine learning ecosystem mirroring the Python scientific stack: **NumPy**, **SciPy**, **scikit-learn**, **scikit-image**, and **numpy-ml**.

Written in Rust 2024 edition, `marray` provides familiar Pythonic APIs, zero external runtime baggage beyond core linear algebra/random primitives, and full type safety and memory safety.

---

## Highlights & Modules

| Module | Description | Python Equivalent |
|---|---|---|
| [`np`](#1-np-numpy-equivalent) | N-dimensional arrays, slicing, linear algebra, FFT, statistical routines, random sampling | `numpy` |
| [`sp`](#2-sp-scipy-equivalent) | Numerical integration, sparse matrices, spatial algorithms, signal filtering, special functions, optimization | `scipy` |
| [`sklearn`](#3-sklearn-scikit-learn-equivalent) | Classical machine learning estimators (trees, ensembles, SVM, neural nets, clustering, PCA, GP, t-SNE) | `scikit-learn` |
| [`skimage`](#4-skimage-scikit-image-equivalent) | Image processing algorithms (filters, edge detection, morphology, color spaces, transformations) | `scikit-image` |
| [`numpy_ml`](#5-numpy_ml-numpy-ml-equivalent) | Deep learning from scratch (Conv1D/2D, BatchNorm, MHA, VAE, Word2Vec), multi-armed bandits, RL agents | `numpy-ml` |

---

## Installation

Add `marray` to your `Cargo.toml`:

```toml
[dependencies]
marray = "1.0.0"
```

### Feature Flags

All features are enabled by default (`default = ["np", "sp", "sklearn", "skimage", "numpy_ml"]`). To enable only specific modules:

```toml
[dependencies]
marray = { version = "1.0.0", default-features = false, features = ["np", "sp", "sklearn"] }
```

---

## Quickstart & Examples

### 1. `np` (NumPy Equivalent)

```rust
use marray::np;

// Create arrays
let a = np::array(vec![1.0, 2.0, 3.0]);
let b: np::Array2<f64> = np::zeros((2, 3));
let c = np::linspace(0.0, 1.0, 5);

// Statistics and vector operations
let m = np::mean(&a);
let s = np::sum(&a);
let clipped = np::clip(&a, 1.5, 2.5);
```

### 2. `sp` (SciPy Equivalent)

```rust
use marray::sp;
use ndarray::Array1;

// Numerical integration (adaptive Simpson quadrature)
let (integral, err) = sp::integrate::quad(|x| x.sin(), 0.0, std::f64::consts::PI);
assert!((integral - 2.0).abs() < 1e-6);

// Sparse matrices and linear solve
let mut coo = sp::sparse::COOMatrix::new(3, 3);
coo.add_entry(0, 0, 4.0);
coo.add_entry(1, 1, 3.0);
coo.add_entry(2, 2, 2.0);
let csr = coo.to_csr();
let b = Array1::from_vec(vec![8.0, 6.0, 4.0]);
let x = sp::sparse::linalg::spsolve(&csr, &b).unwrap(); // [2.0, 2.0, 2.0]

// Spatial algorithms (KD-Tree & Convex Hull)
let pts = ndarray::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
let tree = sp::spatial::KDTree::new(&pts).unwrap();
let (dist, idx) = tree.query(&ndarray::array![0.1, 0.1]).unwrap();

// Special functions
let j0_val = sp::special::jv(0.0, 2.4048255577); // Bessel J_0
let norm_cdf = sp::special::ndtr(0.0);           // Normal CDF (0.5)
```

### 3. `sklearn` (Scikit-Learn Equivalent)

```rust
use marray::sklearn::manifold::TSNE;
use marray::sklearn::mixture::GaussianMixture;
use marray::sklearn::gaussian_process::GaussianProcessRegressor;
use marray::sklearn::feature_selection::SelectKBest;
use ndarray::array;

// Gaussian Mixture Model
let x = array![[0.0, 0.1], [0.1, 0.0], [10.0, 10.1], [10.1, 10.0]];
let mut gmm = GaussianMixture::new(2, 50, 1e-4, "full", Some(42));
gmm.fit(&x).unwrap();
let labels = gmm.predict(&x);

// Dimensionality Reduction (t-SNE)
let mut tsne = TSNE::new(2, 30.0, 200.0, 500, Some(42));
let embedded = tsne.fit_transform(&x).unwrap();
```

### 4. `skimage` (Scikit-Image Equivalent)

```rust
use marray::skimage::filters::{gaussian, threshold_otsu};
use ndarray::Array2;

let img = Array2::<f64>::zeros((64, 64));
let smoothed = gaussian(&img, 1.5);
```

### 5. `numpy_ml` (Numpy-ML Equivalent)

```rust
use marray::numpy_ml::neural_nets::layers::{BatchNorm, Conv1D, Linear, MultiHeadAttention};
use marray::numpy_ml::neural_nets::models::BernoulliVAE;
use marray::numpy_ml::trees::RandomForest;
use marray::numpy_ml::bandits::{BernoulliBandit, EpsilonGreedy, BanditPolicy};
use marray::numpy_ml::rl_models::{QLearningAgent, SARSAAgent};

// Multi-Head Attention layer
let mut mha = MultiHeadAttention::new(64, 8); // d_model=64, n_heads=8

// Variational Autoencoder
let mut vae = BernoulliVAE::new(784, 16, 128); // 784 inputs, 16 latent, 128 hidden

// Multi-Armed Bandits
let bandit = BernoulliBandit::new(vec![0.1, 0.8, 0.3]);
let mut policy = EpsilonGreedy::new(0.1, 0.0);
let (reward, arm) = policy.act(&bandit);
```

---

## Examples & Demos

Run any of the included example scripts:

```bash
# NumPy utilities and array manipulation
cargo run --example numpy_demo

# SciPy algorithms (linalg, stats, signal, fft, interpolation)
cargo run --example scipy_demo

# Scikit-Learn estimators, transformers, and pipelines
cargo run --example sklearn_demo

# Scikit-Image filtering, edges, morphology
cargo run --example skimage_demo

# Numpy-ML neural network layers, VAE, trees, bandits, RL
cargo run --example numpy_ml_demo
```

---

## Testing

Run the full unit test suite (440+ tests):

```bash
cargo test
```

To run tests with release optimizations:

```bash
cargo test --release
```

---

## License

Dual-licensed under either:
- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
