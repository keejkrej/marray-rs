use marray::np;

fn main() {
    println!("--- NumPy Demo ---");

    // Python: a = np.array([1.0, 2.0, 3.0])
    let a = np::array(vec![1.0, 2.0, 3.0]);
    println!("a (1D Array):\n{:?}\n", a);

    // Python: b = np.zeros((2, 3))
    let b: np::Array2<f64> = np::zeros((2, 3));
    println!("b (2D Zeros Array):\n{:?}\n", b);

    // Python: c = np.arange(0, 10, 2)
    let c = np::arange(0.0, 10.0, 2.0);
    println!("c (Arange):\n{:?}\n", c);

    // Python: d = np.linspace(0, 1, 5)
    let d = np::linspace(0.0, 1.0, 5);
    println!("d (Linspace):\n{:?}\n", d);

    // Python: e = a + c[:3]
    // Slicing in Rust: use np::s! macro
    let e = &a + &c.slice(np::s![0..3]);
    println!("e (a + c[:3]):\n{:?}\n", e);

    // Python: print(np.mean(e))
    println!("mean(e): {}", np::mean(&e));
    println!("sum(e): {}", np::sum(&e));
    println!();

    // Python: median_val = np.median(a)
    let a_ext = np::array(vec![1.0, 5.0, 3.0, 4.0, 2.0]);
    println!("median(a_ext): {}", np::median(&a_ext));

    // Python: cov_matrix = np.cov(m)
    let m = np::array(vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0]
    ]);
    println!("cov(m):\n{:?}", np::cov(&m).unwrap());
    println!();

    // Python: clipped = np.clip(a, 2.0, 4.0)
    let vals = np::array(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let clipped = np::clip(&vals, 2.0, 4.0);
    println!("clip([1,2,3,4,5], 2, 4): {:?}", clipped);

    // Python: result = np.where(cond, x, y)
    let cond = np::array(vec![true, false, true, false]);
    let x_w = np::array(vec![10.0, 20.0, 30.0, 40.0]);
    let y_w = np::array(vec![1.0, 2.0, 3.0, 4.0]);
    let w = np::where_arr(&cond, &x_w, &y_w);
    println!("where([T,F,T,F], x, y): {:?}", w);

    // Python: np.unique(arr)
    let dup = np::array(vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0]);
    println!("unique([3,1,2,1,3,2]): {:?}", np::unique(&dup));

    // Python: np.percentile(arr, 25)
    let p_arr = np::array(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    println!("percentile([10..50], 25): {}", np::percentile(&p_arr, 25.0));
    println!("percentile([10..50], 50): {}", np::percentile(&p_arr, 50.0));
    println!("percentile([10..50], 75): {}", np::percentile(&p_arr, 75.0));

    let sorted = np::rssort(&np::array(vec![3.0, 1.0, 2.0]));
    println!("rssort([3,1,2]): {:?}", sorted);
    println!("argsort([3,1,2]): {:?}", np::argsort(&np::array(vec![3.0, 1.0, 2.0])));
    println!("cumsum([1,2,3]): {:?}", np::cumsum(&np::array(vec![1.0, 2.0, 3.0])));
    println!("diff([5,9,7], 1): {:?}", np::diff(&np::array(vec![5.0, 9.0, 7.0]), 1));
    println!("tile([1,2], 2): {:?}", np::tile(&np::array(vec![1.0, 2.0]), 2));
    println!("repeat([1,2], 2): {:?}", np::repeat(&np::array(vec![1.0, 2.0]), 2));
    println!("rsnorm([3,4]): {}", np::linalg::rsnorm(&np::array(vec![3.0, 4.0]), Some(2)));
    println!("cross([1,0,0],[0,1,0]): {:?}", np::cross(&np::array(vec![1.0, 0.0, 0.0]), &np::array(vec![0.0, 1.0, 0.0])));
}
