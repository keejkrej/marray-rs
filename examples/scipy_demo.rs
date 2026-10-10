use marray::{np, sp};

fn main() {
    println!("--- SciPy Demo ---");

    // Linalg solving: A * x = B
    // Python: A = np.array([[4, 7], [2, 6]]); B = np.array([1, 2])
    let a = np::array(vec![vec![4.0, 7.0], vec![2.0, 6.0]]);
    let b = np::array(vec![1.0, 2.0]);

    // Python: x = scipy.linalg.solve(A, B)
    let x = sp::linalg::solve_vec(&a, &b).unwrap();
    println!("x (solution to A*x = B):\n{:?}\n", x);

    // Inverse of a matrix
    // Python: a_inv = scipy.linalg.inv(A)
    let a_inv = sp::linalg::inv(&a).unwrap();
    println!("A^-1 (Inverse):\n{:?}\n", a_inv);

    // FFT demo
    // Python: sig = np.array([1.0, 2.0, 1.0, 2.0])
    //         sig_fft = scipy.fft.fft(sig)
    let sig = np::array(vec![1.0, 2.0, 1.0, 2.0]);
    let sig_fft = sp::fft::fft_real(&sig);
    println!("FFT of signal:\n{:?}\n", sig_fft);

    // Stats demo (Normal distribution pdf/cdf)
    // Python: p = scipy.stats.norm.cdf(0, loc=0, scale=1)
    let p = sp::stats::Norm::cdf(0.0, 0.0, 1.0);
    println!("CDF of standard normal at 0.0: {}\n", p);

    // Correlation demo
    // Python: r, p_val = scipy.stats.pearsonr(x, y)
    let x_arr = np::array(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_arr = np::array(vec![2.0, 4.1, 5.9, 8.0, 10.1]);
    let (r, p_val) = sp::stats::pearsonr(&x_arr, &y_arr).unwrap();
    println!("Pearson correlation: r = {}, p-value = {}\n", r, p_val);

    let (sr, s_p_val) = sp::stats::spearmanr(&x_arr, &y_arr).unwrap();
    println!("Spearman rank correlation: rho = {}, p-value = {}\n", sr, s_p_val);

    // Interpolation demo
    // Python: f = scipy.interpolate.interp1d(x, y)
    let interp_x = np::array(vec![0.0, 1.0, 2.0]);
    let interp_y = np::array(vec![0.0, 10.0, 20.0]);
    let f = sp::interpolate::Interp1D::new(&interp_x, &interp_y).unwrap();
    let x_new = np::array(vec![0.5, 1.5]);
    println!("Interpolated values at [0.5, 1.5]:\n{:?}\n", f.call(&x_new));

    // Digital filter demo (lfilter)
    // Python: y = scipy.signal.lfilter(b, a, x)
    let filter_b = np::array(vec![1.0, 0.5]);
    let filter_a = np::array(vec![1.0, -0.2]);
    let filter_x = np::array(vec![1.0, 2.0, 3.0]);
    let filter_y = sp::signal::lfilter(&filter_b, &filter_a, &filter_x).unwrap();
    println!("Filtered signal (lfilter):\n{:?}\n", filter_y);

    // Signal convolution
    // Python: out = scipy.signal.convolve(in1, in2, mode='full')
    let in1 = np::array(vec![1.0, 2.0, 3.0]);
    let in2 = np::array(vec![0.0, 1.0, 0.5]);
    let out = sp::signal::convolve(&in1, &in2, sp::signal::ConvolveMode::Full);
    println!("Convolved output:\n{:?}\n", out);

    // Z-score demo
    // Python: z = scipy.stats.zscore(x)
    let z_arr = np::array(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let z = sp::stats::zscore(&z_arr, 0.0);
    println!("zscore([1,2,3,4,5], ddof=0):\n{:?}\n", z);

    // Zero-phase filtering demo (filtfilt)
    // Python: y = scipy.signal.filtfilt(b, a, x)
    let ff_b = np::array(vec![1.0, 0.5]);
    let ff_a = np::array(vec![1.0, -0.2]);
    let ff_x = np::array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let ff_y = sp::signal::filtfilt(&ff_b, &ff_a, &ff_x).unwrap();
    println!("filtfilt output:\n{:?}\n", ff_y);

    // Peak finding demo
    // Python: peaks, _ = scipy.signal.find_peaks(x, height=1.5, distance=3)
    let peak_sig = np::array(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.5, 0.0]);
    let peaks = sp::signal::find_peaks(&peak_sig, None, None);
    println!("find_peaks (no filter): {:?}", peaks);
    let peaks_h = sp::signal::find_peaks(&peak_sig, Some(1.5), None);
    println!("find_peaks (height>=1.5): {:?}", peaks_h);
    let peaks_d = sp::signal::find_peaks(&peak_sig, None, Some(3));
    println!("find_peaks (distance>=3): {:?}", peaks_d);

    println!("rsminimize_scalar: {}", sp::optimize::rsminimize_scalar(|x| (x - 2.0).powi(2), (0.0, 4.0), 1e-4));
    let fit = sp::optimize::curve_fit(|params, x| params[0] * x + params[1], &np::array(vec![0.0, 1.0, 2.0]), &np::array(vec![1.0, 3.0, 5.0]), &np::array(vec![1.0, 1.0])).unwrap();
    println!("curve_fit params: {:?}", fit);
    let xa = np::array(vec![vec![0.0, 0.0], vec![1.0, 1.0]]);
    let xb = np::array(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
    println!("rscdist: {:?}", sp::spatial::rscdist(&xa, &xb, "euclidean"));
    println!("rsuniform_filter1d: {:?}", sp::ndimage::rsuniform_filter1d(&np::array(vec![1.0, 2.0, 3.0, 4.0]), 2));
}
