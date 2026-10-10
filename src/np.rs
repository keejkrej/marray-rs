pub use ndarray::{Array, Array1, Array2, Array3, ArrayD, Axis, Dimension, Ix1, Ix2, Ix3, IxDyn};
pub use ndarray::s;

// Re-expose standard ndarray types for convenience
pub type ArrayN<T, D> = Array<T, D>;

/// Helper trait to construct Array from various types of inputs (vecs, slices, nested vecs)
pub trait IntoArray<T, D: Dimension> {
    fn into_array(self) -> Array<T, D>;
}

impl<T> IntoArray<T, Ix1> for Vec<T> {
    fn into_array(self) -> Array<T, Ix1> {
        Array1::from_vec(self)
    }
}

impl<T: Clone> IntoArray<T, Ix1> for &[T] {
    fn into_array(self) -> Array<T, Ix1> {
        Array1::from_vec(self.to_vec())
    }
}

impl<T: Clone, const N: usize> IntoArray<T, Ix1> for [T; N] {
    fn into_array(self) -> Array<T, Ix1> {
        Array1::from_vec(self.to_vec())
    }
}

// 2D Array from nested Vecs
impl<T> IntoArray<T, Ix2> for Vec<Vec<T>> {
    fn into_array(self) -> Array<T, Ix2> {
        let nrows = self.len();
        if nrows == 0 {
            return Array2::from_shape_vec((0, 0), vec![]).unwrap();
        }
        let ncols = self[0].len();
        let mut flat = Vec::with_capacity(nrows * ncols);
        for mut row in self {
            assert_eq!(row.len(), ncols, "All rows must have the same length");
            flat.append(&mut row);
        }
        Array2::from_shape_vec((nrows, ncols), flat).unwrap()
    }
}

// 2D Array from slice of slices
impl<T: Clone> IntoArray<T, Ix2> for &[&[T]] {
    fn into_array(self) -> Array<T, Ix2> {
        let nrows = self.len();
        if nrows == 0 {
            return Array2::from_shape_vec((0, 0), vec![]).unwrap();
        }
        let ncols = self[0].len();
        let mut flat = Vec::with_capacity(nrows * ncols);
        for row in self {
            assert_eq!(row.len(), ncols, "All rows must have the same length");
            flat.extend_from_slice(row);
        }
        Array2::from_shape_vec((nrows, ncols), flat).unwrap()
    }
}

// Identity conversion for existing arrays
impl<T, D: Dimension> IntoArray<T, D> for Array<T, D> {
    fn into_array(self) -> Array<T, D> {
        self
    }
}

/// Creates a NumPy-like array.
/// Works with vectors, slices, arrays, and nested structures.
pub fn array<T, D: Dimension>(data: impl IntoArray<T, D>) -> Array<T, D> {
    data.into_array()
}

/// Create an array of zeros.
pub fn zeros<T: Clone + num_traits::Zero, Sh>(shape: Sh) -> Array<T, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    Array::zeros(shape)
}

/// Create an array of ones.
pub fn ones<T: Clone + num_traits::One, Sh>(shape: Sh) -> Array<T, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    Array::from_elem(shape, T::one())
}

/// Create an identity matrix.
pub fn eye<T: Clone + num_traits::Zero + num_traits::One>(n: usize) -> Array2<T> {
    Array2::eye(n)
}

/// Create an array filled with zeros with the same shape as the input array.
pub fn zeros_like<T: Clone + num_traits::Zero, D: Dimension>(arr: &Array<T, D>) -> Array<T, D> {
    Array::zeros(arr.raw_dim())
}

/// Create an array filled with ones with the same shape as the input array.
pub fn ones_like<T: Clone + num_traits::One, D: Dimension>(arr: &Array<T, D>) -> Array<T, D> {
    Array::from_elem(arr.raw_dim(), T::one())
}

/// Create a sequence of numbers from start to stop with a given step.
pub fn arange<T>(start: T, stop: T, step: T) -> Array1<T>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    let mut values = Vec::new();
    let mut curr = start;
    let step_f = step.to_f64().unwrap();
    if step_f > 0.0 {
        while curr < stop {
            values.push(curr);
            let next_f = curr.to_f64().unwrap() + step_f;
            curr = T::from_f64(next_f).unwrap_or(curr);
        }
    } else if step_f < 0.0 {
        while curr > stop {
            values.push(curr);
            let next_f = curr.to_f64().unwrap() + step_f;
            curr = T::from_f64(next_f).unwrap_or(curr);
        }
    }
    Array1::from_vec(values)
}

/// Create an array of num evenly spaced values from start to stop.
pub fn linspace<T>(start: T, stop: T, num: usize) -> Array1<T>
where
    T: num_traits::Float + num_traits::FromPrimitive,
{
    if num == 0 {
        return Array1::from_vec(vec![]);
    }
    if num == 1 {
        return Array1::from_vec(vec![start]);
    }
    let step = (stop - start) / T::from_usize(num - 1).unwrap();
    Array1::from_shape_fn(num, |i| start + T::from_usize(i).unwrap() * step)
}

pub mod random;

#[cfg(feature = "sp")]
pub mod linalg {
    pub use crate::sp::linalg::*;

    use ndarray::{ArrayBase, Data, Dimension};

    /// Vector/matrix norm helper for L1/L2/Frobenius-like norms.
    pub fn rsnorm<S, D>(arr: &ArrayBase<S, D>, ord: Option<usize>) -> f64
    where
        S: Data<Elem = f64>,
        D: Dimension,
    {
        let ord = ord.unwrap_or(2);
        let values: Vec<f64> = arr.iter().copied().collect();
        match ord {
            0 => values.iter().filter(|&&v| v.abs() > 0.0).count() as f64,
            1 => values.iter().map(|v| v.abs()).sum(),
            2 => values.iter().map(|v| v * v).sum::<f64>().sqrt(),
            _ => values.iter().map(|v| v.abs()).fold(0.0, |a, b| if a > b { a } else { b }),
        }
    }
}


// Element-wise Math

pub fn sin<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.sin())
}

pub fn cos<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.cos())
}

pub fn exp<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.exp())
}

pub fn log<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.ln())
}

pub fn sqrt<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.sqrt())
}

pub fn abs<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Signed + Clone,
    D: Dimension,
{
    arr.mapv(|x| x.abs())
}

// Reductions

pub fn sum<T, D>(arr: &Array<T, D>) -> T
where
    T: Clone + num_traits::Zero + std::ops::Add<Output = T>,
    D: Dimension,
{
    arr.sum()
}

pub fn sum_axis<T, D>(arr: &Array<T, D>, axis: usize) -> Array<T, <D as Dimension>::Smaller>
where
    T: Clone + num_traits::Zero + std::ops::Add<Output = T>,
    D: Dimension + ndarray::RemoveAxis,
{
    arr.sum_axis(Axis(axis))
}

pub fn mean<T, D>(arr: &Array<T, D>) -> T
where
    T: num_traits::Float + num_traits::FromPrimitive,
    D: Dimension,
{
    arr.mean().unwrap_or_else(|| T::nan())
}

pub fn mean_axis<T, D>(arr: &Array<T, D>, axis: usize) -> Array<T, <D as Dimension>::Smaller>
where
    T: num_traits::Float + num_traits::FromPrimitive,
    D: Dimension + ndarray::RemoveAxis,
{
    arr.mean_axis(Axis(axis)).unwrap()
}

pub fn var<T, D>(arr: &Array<T, D>, ddof: f64) -> T
where
    T: num_traits::Float + num_traits::FromPrimitive,
    D: Dimension,
{
    let ddof_t = T::from_f64(ddof).unwrap_or_else(T::zero);
    arr.var(ddof_t)
}

pub fn std<T, D>(arr: &Array<T, D>, ddof: f64) -> T
where
    T: num_traits::Float + num_traits::FromPrimitive,
    D: Dimension,
{
    let ddof_t = T::from_f64(ddof).unwrap_or_else(T::zero);
    arr.std(ddof_t)
}

pub fn min<T, D>(arr: &Array<T, D>) -> T
where
    T: Clone + PartialOrd,
    D: Dimension,
{
    arr.iter()
        .fold(arr.first().unwrap().clone(), |a, b| if a < *b { a } else { b.clone() })
}

pub fn max<T, D>(arr: &Array<T, D>) -> T
where
    T: Clone + PartialOrd,
    D: Dimension,
{
    arr.iter()
        .fold(arr.first().unwrap().clone(), |a, b| if a > *b { a } else { b.clone() })
}

pub fn argmin<T, D>(arr: &Array<T, D>) -> usize
where
    T: PartialOrd,
    D: Dimension,
{
    let mut min_idx = 0;
    let mut min_val = arr.first().unwrap();
    for (i, val) in arr.iter().enumerate() {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    min_idx
}

pub fn argmax<T, D>(arr: &Array<T, D>) -> usize
where
    T: PartialOrd,
    D: Dimension,
{
    let mut max_idx = 0;
    let mut max_val = arr.first().unwrap();
    for (i, val) in arr.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

// Manipulations

pub fn transpose<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: Clone,
    D: Dimension,
{
    arr.t().to_owned()
}

pub fn reshape<T, D, Sh>(arr: &Array<T, D>, shape: Sh) -> Array<T, Sh::Dim>
where
    T: Clone,
    D: Dimension,
    Sh: ndarray::IntoDimension,
{
    arr.to_shape(shape).expect("Shape dimensions must match input array size").to_owned()
}

pub fn concatenate<T, D>(arrays: &[&Array<T, D>], axis: usize) -> Array<T, D>
where
    T: Clone,
    D: Dimension + ndarray::RemoveAxis,
{
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    ndarray::concatenate(Axis(axis), &views).expect("Failed to concatenate arrays")
}

pub fn stack<T, D>(arrays: &[&Array<T, D>], axis: usize) -> Array<T, D::Larger>
where
    T: Clone,
    D: Dimension + ndarray::RemoveAxis,
    D::Larger: Dimension,
{
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    ndarray::stack(Axis(axis), &views).expect("Failed to stack arrays")
}

// Linear Algebra

pub fn dot<T>(a: &Array1<T>, b: &Array1<T>) -> T
where
    T: Clone + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    assert_eq!(a.len(), b.len(), "Array lengths must match for dot product");
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + a[i].clone() * b[i].clone();
    }
    sum
}

pub fn matmul<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: Clone + num_traits::Zero + num_traits::One + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + ndarray::LinalgScalar,
{
    a.dot(b)
}

/// Cross product of two 3-element vectors.
pub fn cross(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    assert_eq!(a.len(), 3, "Cross product expects 3-element vectors");
    assert_eq!(b.len(), 3, "Cross product expects 3-element vectors");
    Array1::from_vec(vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

/// Return a sorted copy of an array.
pub fn rssort<T>(arr: &Array1<T>) -> Array1<T>
where
    T: Clone + PartialOrd,
{
    let mut values = arr.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Array1::from_vec(values)
}

/// Return the indices that would sort an array.
pub fn argsort<T>(arr: &Array1<T>) -> Array1<usize>
where
    T: PartialOrd,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(|&a, &b| arr[a].partial_cmp(&arr[b]).unwrap_or(std::cmp::Ordering::Equal));
    Array1::from_vec(indices)
}

/// Compute the cumulative sum along a 1D array.
pub fn cumsum<T>(arr: &Array1<T>) -> Array1<T>
where
    T: Clone + num_traits::Zero + std::ops::Add<Output = T>,
{
    let mut out = Vec::with_capacity(arr.len());
    let mut running = T::zero();
    for value in arr.iter().cloned() {
        running = running + value;
        out.push(running.clone());
    }
    Array1::from_vec(out)
}

/// Compute the n-th discrete difference along a 1D array.
pub fn diff<T>(arr: &Array1<T>, n: usize) -> Array1<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    if n == 0 || arr.len() <= 1 {
        return arr.clone();
    }

    let mut values = arr.to_vec();
    for _ in 0..n {
        if values.len() < 2 {
            return Array1::from_vec(Vec::new());
        }
        let mut next = Vec::with_capacity(values.len() - 1);
        for i in 1..values.len() {
            next.push(values[i].clone() - values[i - 1].clone());
        }
        values = next;
    }
    Array1::from_vec(values)
}

/// Repeat an array reps times.
pub fn tile<T>(arr: &Array1<T>, reps: usize) -> Array1<T>
where
    T: Clone,
{
    let mut out = Vec::with_capacity(arr.len() * reps);
    for _ in 0..reps {
        out.extend_from_slice(&arr.to_vec());
    }
    Array1::from_vec(out)
}

/// Repeat each element of an array n times.
pub fn repeat<T>(arr: &Array1<T>, n: usize) -> Array1<T>
where
    T: Clone,
{
    let mut out = Vec::with_capacity(arr.len() * n);
    for value in arr.iter().cloned() {
        for _ in 0..n {
            out.push(value.clone());
        }
    }
    Array1::from_vec(out)
}

/// Compute the median along a 1D array.
pub fn median(arr: &Array1<f64>) -> f64 {
    let n = arr.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut sorted = arr.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Compute covariance matrix of observations.
pub fn cov(m: &Array2<f64>) -> Result<Array2<f64>, String> {
    let (nrows, ncols) = m.dim();
    if ncols <= 1 {
        return Err("Covariance requires at least 2 observations (columns)".to_string());
    }

    let row_means = m.mean_axis(Axis(1)).unwrap();

    let mut cov_matrix = Array2::zeros((nrows, nrows));
    for i in 0..nrows {
        for k in 0..nrows {
            let mut sum = 0.0;
            for j in 0..ncols {
                sum += (m[[i, j]] - row_means[i]) * (m[[k, j]] - row_means[k]);
            }
            cov_matrix[[i, k]] = sum / ((ncols - 1) as f64);
        }
    }
    Ok(cov_matrix)
}

/// Clip (limit) the values in an array.
pub fn clip<T, D>(arr: &Array<T, D>, min: T, max: T) -> Array<T, D>
where
    T: Clone + PartialOrd,
    D: Dimension,
{
    arr.mapv(|val| {
        if val < min {
            min.clone()
        } else if val > max {
            max.clone()
        } else {
            val
        }
    })
}

/// Return elements chosen from x or y depending on condition.
pub fn where_arr<T, D>(cond: &Array<bool, D>, x: &Array<T, D>, y: &Array<T, D>) -> Array<T, D>
where
    T: Clone,
    D: Dimension,
{
    assert_eq!(cond.shape(), x.shape(), "Condition and x must have the same shape");
    assert_eq!(cond.shape(), y.shape(), "Condition and y must have the same shape");
    let flat_vec: Vec<T> = cond
        .iter()
        .zip(x.iter())
        .zip(y.iter())
        .map(|((&c, vx), vy)| if c { vx.clone() } else { vy.clone() })
        .collect();
    Array::from_shape_vec(x.raw_dim(), flat_vec).unwrap()
}

/// Find the unique elements of an array.
pub fn unique<T>(arr: &Array1<T>) -> Array1<T>
where
    T: Clone + PartialOrd,
{
    let mut vec = arr.to_vec();
    vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vec.dedup_by(|a, b| {
        let ar: &T = a;
        let br: &T = b;
        ar.partial_cmp(br).unwrap_or(std::cmp::Ordering::Equal) == std::cmp::Ordering::Equal
    });
    Array1::from_vec(vec)
}

/// Compute the q-th percentile of the data.
/// q must be in range [0.0, 100.0].
pub fn percentile(arr: &Array1<f64>, q: f64) -> f64 {
    assert!((0.0..=100.0).contains(&q), "q must be between 0.0 and 100.0");
    let n = arr.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut sorted = arr.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if n == 1 {
        return sorted[0];
    }

    let idx = (q / 100.0) * ((n - 1) as f64);
    let low = idx.floor() as usize;
    let high = idx.ceil() as usize;
    if low == high {
        sorted[low]
    } else {
        let weight = idx - (low as f64);
        sorted[low] * (1.0 - weight) + sorted[high] * weight
    }
}

/// Return a new array of given shape and type, filled with fill_value.
pub fn full<T: Clone, Sh>(shape: Sh, fill_value: T) -> Array<T, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    Array::from_elem(shape, fill_value)
}

/// Return a full array with the same shape and type as a given array.
pub fn full_like<T: Clone, D: Dimension>(arr: &Array<T, D>, fill_value: T) -> Array<T, D> {
    Array::from_elem(arr.raw_dim(), fill_value)
}

/// Return numbers spaced evenly on a log scale.
pub fn logspace(start: f64, stop: f64, num: usize, base: f64) -> Array1<f64> {
    let exponents = linspace(start, stop, num);
    exponents.mapv(|exp| base.powf(exp))
}

/// Return numbers spaced evenly on a log scale (geometric progression).
pub fn geomspace(start: f64, stop: f64, num: usize) -> Array1<f64> {
    assert!(start > 0.0 && stop > 0.0, "geomspace bounds must be strictly positive");
    let log_start = start.ln();
    let log_stop = stop.ln();
    let lin = linspace(log_start, log_stop, num);
    lin.mapv(|x| x.exp())
}

/// Returns True if two arrays are element-wise equal within a tolerance.
pub fn allclose<D: Dimension>(a: &Array<f64, D>, b: &Array<f64, D>, rtol: f64, atol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(&x, &y)| {
        if x.is_nan() || y.is_nan() {
            return false;
        }
        (x - y).abs() <= (atol + rtol * y.abs())
    })
}

/// Returns a boolean array where two arrays are element-wise equal within a tolerance.
pub fn isclose<D: Dimension>(a: &Array<f64, D>, b: &Array<f64, D>, rtol: f64, atol: f64) -> Array<bool, D> {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for isclose");
    let mut out = Array::from_elem(a.raw_dim(), false);
    for ((&x, &y), out_val) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        *out_val = (x - y).abs() <= (atol + rtol * y.abs());
    }
    out
}

/// Test whether all array elements evaluate to True.
pub fn all<D: Dimension>(arr: &Array<bool, D>) -> bool {
    arr.iter().all(|&v| v)
}

/// Test whether any array elements evaluate to True.
pub fn any<D: Dimension>(arr: &Array<bool, D>) -> bool {
    arr.iter().any(|&v| v)
}

/// Extract a diagonal or construct a diagonal array.
pub fn diag(arr: &Array2<f64>, k: i32) -> Array1<f64> {
    let (nrows, ncols) = arr.dim();
    let mut diag_vals = Vec::new();
    for i in 0..nrows {
        let j = (i as i32) + k;
        if j >= 0 && (j as usize) < ncols {
            diag_vals.push(arr[[i, j as usize]]);
        }
    }
    Array1::from_vec(diag_vals)
}

/// Create a two-dimensional array with the flattened input as a diagonal.
pub fn diagflat(v: &Array1<f64>, k: i32) -> Array2<f64> {
    let n = v.len();
    let offset = k.abs() as usize;
    let size = n + offset;
    let mut out = Array2::zeros((size, size));
    for (idx, &val) in v.iter().enumerate() {
        if k >= 0 {
            out[[idx, idx + offset]] = val;
        } else {
            out[[idx + offset, idx]] = val;
        }
    }
    out
}

/// Return the sum along diagonals of the array.
pub fn trace(arr: &Array2<f64>) -> f64 {
    let min_dim = arr.nrows().min(arr.ncols());
    let mut sum = 0.0;
    for i in 0..min_dim {
        sum += arr[[i, i]];
    }
    sum
}

/// Compute the outer product of two vectors.
pub fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut out = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[[i, j]] = a[i] * b[j];
        }
    }
    out
}

/// Inner product of two arrays.
pub fn inner(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    dot(a, b)
}

/// Return Pearson product-moment correlation coefficients.
pub fn corrcoef(m: &Array2<f64>) -> Result<Array2<f64>, String> {
    let c = cov(m)?;
    let (n, _) = c.dim();
    let mut r = Array2::zeros((n, n));
    let diag_std: Vec<f64> = (0..n).map(|i| c[[i, i]].sqrt()).collect();
    for i in 0..n {
        for j in 0..n {
            let denom = diag_std[i] * diag_std[j];
            if denom > 1e-12 {
                r[[i, j]] = (c[[i, j]] / denom).clamp(-1.0, 1.0);
            } else {
                r[[i, j]] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    Ok(r)
}

/// Roll array elements along a given axis.
pub fn roll(arr: &Array1<f64>, shift: isize) -> Array1<f64> {
    let n = arr.len();
    if n == 0 {
        return arr.clone();
    }
    let shift_norm = ((shift % (n as isize)) + (n as isize)) as usize % n;
    let mut out = Array1::zeros(n);
    for i in 0..n {
        let new_idx = (i + shift_norm) % n;
        out[new_idx] = arr[i];
    }
    out
}

/// Reverse the order of elements in an array.
pub fn flip(arr: &Array1<f64>) -> Array1<f64> {
    let mut v = arr.to_vec();
    v.reverse();
    Array1::from_vec(v)
}

/// Pad an array with constant values.
pub fn pad(arr: &Array1<f64>, pad_width: (usize, usize), constant_value: f64) -> Array1<f64> {
    let (left, right) = pad_width;
    let mut v = Vec::with_capacity(left + arr.len() + right);
    v.extend(std::iter::repeat_n(constant_value, left));
    v.extend(arr.iter().copied());
    v.extend(std::iter::repeat_n(constant_value, right));
    Array1::from_vec(v)
}

/// Remove single-dimensional entries from the shape of an array.
pub fn squeeze(arr: &Array2<f64>) -> Array1<f64> {
    let (r, c) = arr.dim();
    if r == 1 {
        arr.row(0).to_owned()
    } else if c == 1 {
        arr.column(0).to_owned()
    } else {
        panic!("squeeze: at least one dimension must be 1, got shape {:?}", arr.dim());
    }
}

/// Return a contiguous flattened 1D array.
pub fn ravel(arr: &Array2<f64>) -> Array1<f64> {
    let mut v = Vec::with_capacity(arr.len());
    for row in arr.rows() {
        v.extend(row.iter().copied());
    }
    Array1::from_vec(v)
}

/// Return a flattened copy of the array.
pub fn flatten(arr: &Array2<f64>) -> Array1<f64> {
    ravel(arr)
}

/// Return a sorted copy of an array (alias for rssort).
pub fn sort<T: Clone + PartialOrd>(arr: &Array1<T>) -> Array1<T> {
    rssort(arr)
}

/// Element-wise maximum of array elements.
pub fn maximum<D: Dimension>(a: &Array<f64, D>, b: &Array<f64, D>) -> Array<f64, D> {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for maximum");
    let mut out = Array::zeros(a.raw_dim());
    for ((&x, &y), out_val) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        *out_val = x.max(y);
    }
    out
}

/// Element-wise minimum of array elements.
pub fn minimum<D: Dimension>(a: &Array<f64, D>, b: &Array<f64, D>) -> Array<f64, D> {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for minimum");
    let mut out = Array::zeros(a.raw_dim());
    for ((&x, &y), out_val) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        *out_val = x.min(y);
    }
    out
}

/// Round elements of the array to the nearest integer.
pub fn round<D: Dimension>(arr: &Array<f64, D>) -> Array<f64, D> {
    arr.mapv(|x| x.round())
}

/// Return the floor of the input, element-wise.
pub fn floor<D: Dimension>(arr: &Array<f64, D>) -> Array<f64, D> {
    arr.mapv(|x| x.floor())
}

/// Return the ceiling of the input, element-wise.
pub fn ceil<D: Dimension>(arr: &Array<f64, D>) -> Array<f64, D> {
    arr.mapv(|x| x.ceil())
}

/// Return coordinate matrices from coordinate vectors.
pub fn meshgrid(x: &Array1<f64>, y: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
    let nx = x.len();
    let ny = y.len();
    let mut xx = Array2::zeros((ny, nx));
    let mut yy = Array2::zeros((ny, nx));
    for j in 0..ny {
        for i in 0..nx {
            xx[[j, i]] = x[i];
            yy[[j, i]] = y[j];
        }
    }
    (xx, yy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let a: Array1<i32> = array(vec![1, 2, 3]);
        assert_eq!(a.len(), 3);
        assert_eq!(a[0], 1);

        let b: Array2<i32> = array(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(b.dim(), (2, 2));
        assert_eq!(b[[1, 0]], 3);
    }

    #[test]
    fn test_zeros_ones() {
        let z: Array2<f64> = zeros((2, 3));
        assert_eq!(z.dim(), (2, 3));
        assert_eq!(z[[1, 2]], 0.0);

        let o: Array1<i32> = ones(4);
        assert_eq!(o.len(), 4);
        assert_eq!(o[3], 1);
    }

    #[test]
    fn test_arange_linspace() {
        let a = arange(0.0, 5.0, 1.0);
        assert_eq!(a.len(), 5);
        assert_eq!(a[4], 4.0);

        let l = linspace(0.0, 1.0, 5);
        assert_eq!(l.len(), 5);
        assert_eq!(l[4], 1.0);
        assert_eq!(l[2], 0.5);
    }

    #[test]
    fn test_math_and_reductions() {
        let a = array(vec![1.0, 2.0, 3.0]);
        assert_eq!(sum(&a), 6.0);
        assert_eq!(mean(&a), 2.0);
        assert_eq!(min(&a), 1.0);
        assert_eq!(max(&a), 3.0);
        assert_eq!(argmin(&a), 0);
        assert_eq!(argmax(&a), 2);
    }

    #[test]
    fn test_median_and_cov() {
        let a = array(vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(median(&a), 2.5);

        let m = array(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0]
        ]);
        let c = cov(&m).unwrap();
        assert_eq!(c.dim(), (2, 2));
        assert!((c[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((c[[0, 1]] - 2.0).abs() < 1e-9);
        assert!((c[[1, 1]] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_numpy_helpers_round_trip() {
        let arr = array(vec![3.0, 1.0, 2.0]);
        assert_eq!(rssort(&arr), array(vec![1.0, 2.0, 3.0]));
        assert_eq!(argsort(&arr), array(vec![1, 2, 0]));
        assert_eq!(cumsum(&arr), array(vec![3.0, 4.0, 6.0]));
        assert_eq!(diff(&arr, 1), array(vec![-2.0, 1.0]));
        assert_eq!(tile(&array(vec![1.0, 2.0]), 2), array(vec![1.0, 2.0, 1.0, 2.0]));
        assert_eq!(repeat(&array(vec![1.0, 2.0]), 2), array(vec![1.0, 1.0, 2.0, 2.0]));
        assert!((linalg::rsnorm(&array(vec![3.0, 4.0]), Some(2)) - 5.0).abs() < 1e-9);
        assert_eq!(cross(&array(vec![1.0, 0.0, 0.0]), &array(vec![0.0, 1.0, 0.0])), array(vec![0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_clip_where_unique_percentile() {
        let a = array(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let clipped = clip(&a, 2.0, 4.0);
        assert_eq!(clipped, array(vec![2.0, 2.0, 3.0, 4.0, 4.0]));

        let cond = array(vec![true, false, true, false, true]);
        let x = array(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let y = array(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let w = where_arr(&cond, &x, &y);
        assert_eq!(w, array(vec![10.0, 2.0, 30.0, 4.0, 50.0]));

        let u_arr = array(vec![3.0, 1.0, 2.0, 1.0, 3.0]);
        let u = unique(&u_arr);
        assert_eq!(u, array(vec![1.0, 2.0, 3.0]));

        let p_arr = array(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(percentile(&p_arr, 50.0), 3.0);
        assert_eq!(percentile(&p_arr, 25.0), 2.0);
        assert_eq!(percentile(&p_arr, 75.0), 4.0);
    }
}

