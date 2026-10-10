use ndarray::Array2;

pub struct SimpleImputer {
    pub strategy: String,
    pub fill_value: f64,
    statistics: Option<Vec<f64>>,
}

impl SimpleImputer {
    pub fn new(strategy: &str, fill_value: f64) -> Self {
        Self {
            strategy: strategy.to_string(),
            fill_value,
            statistics: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let (_nrows, ncols) = x.dim();
        let mut stats = Vec::with_capacity(ncols);

        for col in 0..ncols {
            let valid: Vec<f64> = x
                .column(col)
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();

            let stat = if valid.is_empty() {
                match self.strategy.as_str() {
                    "constant" => self.fill_value,
                    _ => f64::NAN,
                }
            } else {
                match self.strategy.as_str() {
                    "mean" => valid.iter().sum::<f64>() / valid.len() as f64,
                    "median" => {
                        let mut sorted = valid.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let n = sorted.len();
                        if n.is_multiple_of(2) {
                            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                        } else {
                            sorted[n / 2]
                        }
                    }
                    "mode" => {
                        let mut counts = std::collections::HashMap::new();
                        for &v in &valid {
                            let key = v.to_bits();
                            *counts.entry(key).or_insert(0usize) += 1;
                        }
                        let mode_bits = counts
                            .into_iter()
                            .max_by_key(|(_, c)| *c)
                            .map(|(bits, _)| bits)
                            .unwrap();
                        f64::from_bits(mode_bits)
                    }
                    "constant" => self.fill_value,
                    _ => panic!("Unknown strategy: {}", self.strategy),
                }
            };

            stats.push(stat);
        }

        self.statistics = Some(stats);
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let stats = self
            .statistics
            .as_ref()
            .expect("SimpleImputer must be fitted before transform");
        let (nrows, ncols) = x.dim();
        let mut out = x.clone();

        for col in 0..ncols {
            for row in 0..nrows {
                if out[[row, col]].is_nan() {
                    out[[row, col]] = stats[col];
                }
            }
        }

        out
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x)
    }
}
