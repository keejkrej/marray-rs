//! # mlab-rs
//!
//! A comprehensive, high-performance, pure-Rust scientific computing and machine learning library
//! providing familiar Pythonic APIs inspired by **NumPy**, **SciPy**, **scikit-learn**, **scikit-image**,
//! and **numpy-ml**.
//!
//! ## Modules
//! - [`np`]: Multidimensional array manipulation, slicing, linear algebra, FFT, statistical routines, and random sampling (NumPy equivalent).
//! - [`sp`]: Numerical algorithms including integration, interpolation, optimization, signal processing, sparse matrices, spatial queries, and special functions (SciPy equivalent).
//! - [`sklearn`]: Classical machine learning estimators for classification, regression, clustering, dimensionality reduction, preprocessing, and model selection (scikit-learn equivalent).
//! - [`skimage`]: Comprehensive 2D/3D image processing algorithms including color conversions, filters, morphology, features, and transforms (scikit-image equivalent).
//! - [`numpy_ml`]: Deep learning and specialized machine learning algorithms from scratch: neural networks (layers, activations, optimizers, schedulers, VAE, Word2Vec), decision trees, multi-armed bandits, reinforcement learning agents, and probabilistic models (numpy-ml equivalent).

#[cfg(feature = "np")]
pub mod np;

#[cfg(feature = "sp")]
pub mod sp;

#[cfg(feature = "sklearn")]
pub mod sklearn;

#[cfg(feature = "skimage")]
pub mod skimage;

#[cfg(feature = "numpy_ml")]
pub mod numpy_ml;
