use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use rand::Rng;
use special::Gamma;

/// Vanilla (non-smoothed) Latent Dirichlet Allocation trained with variational EM.
pub struct LDA {
    pub n_topics: usize,
    pub alpha: Option<Array1<f64>>,
    pub beta: Option<Array2<f64>>,
    pub phi: Vec<Array2<f64>>,
    pub gamma: Option<Array2<f64>>,
    pub corpus: Vec<Vec<usize>>,
    pub n_docs: usize,
    pub n_words_per_doc: Vec<usize>,
    pub vocab_size: usize,
}

impl LDA {
    /// Create a new LDA model with `n_topics` topics.
    pub fn new(n_topics: usize) -> Self {
        Self {
            n_topics,
            alpha: None,
            beta: None,
            phi: Vec::new(),
            gamma: None,
            corpus: Vec::new(),
            n_docs: 0,
            n_words_per_doc: Vec::new(),
            vocab_size: 0,
        }
    }

    fn maximize_phi(&mut self) {
        let d = self.n_docs;
        let t = self.n_topics;
        let beta = self.beta.as_ref().unwrap();
        let gamma = self.gamma.as_ref().unwrap();
        let corpus = &self.corpus;

        for doc in 0..d {
            let n = self.n_words_per_doc[doc];
            for word in 0..n {
                let w_n = corpus[doc][word];
                let mut sum = 0.0;
                for topic in 0..t {
                    let val = beta[[w_n, topic]] * dg(gamma, doc, topic).exp();
                    self.phi[doc][[word, topic]] = val;
                    sum += val;
                }
                for topic in 0..t {
                    self.phi[doc][[word, topic]] /= sum;
                }
            }
        }
    }

    fn maximize_gamma(&mut self) {
        let d = self.n_docs;
        let t = self.n_topics;
        let alpha = self.alpha.as_ref().unwrap();
        let phi = &self.phi;

        let mut gamma = Array2::<f64>::zeros((d, t));
        for doc in 0..d {
            for topic in 0..t {
                let mut sum = 0.0;
                for word in 0..self.n_words_per_doc[doc] {
                    sum += phi[doc][[word, topic]];
                }
                gamma[[doc, topic]] = alpha[topic] + sum;
            }
        }
        self.gamma = Some(gamma);
    }

    fn maximize_beta(&mut self) {
        let t = self.n_topics;
        let v = self.vocab_size;
        let phi = &self.phi;
        let corpus = &self.corpus;

        let mut beta = Array2::<f64>::zeros((v, t));
        for word in 0..v {
            for topic in 0..t {
                let mut sum = 0.0;
                for doc in 0..self.n_docs {
                    for n in 0..self.n_words_per_doc[doc] {
                        if corpus[doc][n] == word {
                            sum += phi[doc][[n, topic]];
                        }
                    }
                }
                beta[[word, topic]] = sum;
            }
        }

        // Normalize columns
        for topic in 0..t {
            let col_sum: f64 = (0..v).map(|word| beta[[word, topic]]).sum();
            for word in 0..v {
                beta[[word, topic]] /= col_sum;
            }
        }

        self.beta = Some(beta);
    }

    fn maximize_alpha(&mut self, max_iters: usize, tol: f64) {
        let d = self.n_docs;
        let t = self.n_topics;
        let gamma = self.gamma.as_ref().unwrap();

        let mut alpha = self.alpha.as_ref().unwrap().clone();
        for _ in 0..max_iters {
            let alpha_old = alpha.clone();

            let gamma_row_sum: Array1<f64> = gamma.sum_axis(ndarray::Axis(1));

            let mut g = Array1::<f64>::zeros(t);
            for topic in 0..t {
                let digamma_sum_alpha = alpha.sum().digamma();
                let mut sum_digamma_gamma = 0.0;
                for doc in 0..d {
                    sum_digamma_gamma += gamma[[doc, topic]].digamma() - gamma_row_sum[doc].digamma();
                }
                g[topic] = d as f64 * (digamma_sum_alpha - alpha[topic].digamma()) + sum_digamma_gamma;
            }

            let mut h = Array1::<f64>::zeros(t);
            for topic in 0..t {
                h[topic] = -(d as f64) * alpha[topic].trigamma();
            }

            let z = d as f64 * alpha.sum().trigamma();

            let mut c_num = 0.0;
            let mut c_den = 0.0;
            for topic in 0..t {
                c_num += g[topic] / h[topic];
                c_den += 1.0 / h[topic];
            }
            let c = c_num / (z.powi(-1) + c_den);

            for topic in 0..t {
                alpha[topic] -= (g[topic] - c) / h[topic];
            }

            let diff: f64 = alpha.iter().zip(alpha_old.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / t as f64;
            if diff.sqrt() < tol {
                break;
            }
        }

        self.alpha = Some(alpha);
    }

    fn e_step(&mut self) {
        self.maximize_phi();
        self.maximize_gamma();
    }

    fn m_step(&mut self) {
        self.maximize_beta();
        self.maximize_alpha(1000, 0.1);
    }

    /// Compute the variational lower bound for the current parameters.
    pub fn vlb(&self) -> f64 {
        let phi = &self.phi;
        let alpha = self.alpha.as_ref().unwrap();
        let beta = self.beta.as_ref().unwrap();
        let gamma = self.gamma.as_ref().unwrap();
        let corpus = &self.corpus;

        let d = self.n_docs;
        let t = self.n_topics;

        let alpha_sum = alpha.sum();
        let alpha_sum_ln_gamma = special::Gamma::ln_gamma(alpha_sum).0;

        let mut a_total = 0.0;
        let mut d_total = 0.0;
        let mut b_total = 0.0;
        let mut c_total = 0.0;

        for doc in 0..d {
            let alpha_term = alpha_sum_ln_gamma - alpha.iter().map(|&a| special::Gamma::ln_gamma(a).0).sum::<f64>();
            let mut alpha_gamma_term = 0.0;
            for topic in 0..t {
                alpha_gamma_term += (alpha[topic] - 1.0) * dg(gamma, doc, topic);
            }
            a_total += alpha_term + alpha_gamma_term;

            let gamma_sum = gamma.row(doc).sum();
            let gamma_term = special::Gamma::ln_gamma(gamma_sum).0 - gamma.row(doc).iter().map(|&g| special::Gamma::ln_gamma(g).0).sum::<f64>();
            let mut gamma_phi_term = 0.0;
            for topic in 0..t {
                gamma_phi_term += (gamma[[doc, topic]] - 1.0) * dg(gamma, doc, topic);
            }
            d_total += gamma_term + gamma_phi_term;

            for n in 0..self.n_words_per_doc[doc] {
                let w_n = corpus[doc][n];
                for topic in 0..t {
                    let phi_val = phi[doc][[n, topic]];
                    b_total += phi_val * dg(gamma, doc, topic);
                    c_total += phi_val * beta[[w_n, topic]].ln();
                    d_total += phi_val * phi_val.ln();
                }
            }
        }

        a_total + b_total + c_total - d_total
    }

    fn initialize_parameters(&mut self) {
        let t = self.n_topics;
        let v = self.vocab_size;
        let d = self.n_docs;

        let mut rng = rand::thread_rng();

        // alpha ~ 100 * Dirichlet(10 * 1_T)
        let dirichlet = rand_distr::Dirichlet::new_with_size(10.0, t).unwrap();
        self.alpha = Some(Array1::from_vec(dirichlet.sample(&mut rng).iter().map(|&x| 100.0 * x).collect()));

        // beta ~ Dirichlet(1_V) for each topic, transposed -> (V, T)
        let mut beta = Array2::<f64>::zeros((v, t));
        let dirichlet_v = rand_distr::Dirichlet::new_with_size(1.0, v).unwrap();
        for topic in 0..t {
            let sample = dirichlet_v.sample(&mut rng);
            for word in 0..v {
                beta[[word, topic]] = sample[word];
            }
        }
        self.beta = Some(beta);

        // phi uniform
        self.phi = Vec::with_capacity(d);
        for doc in 0..d {
            let n = self.n_words_per_doc[doc];
            self.phi.push(Array2::from_elem((n, t), 1.0 / t as f64));
        }

        // gamma = alpha + N_d / T
        let alpha = self.alpha.as_ref().unwrap();
        let mut gamma = Array2::<f64>::zeros((d, t));
        for doc in 0..d {
            for topic in 0..t {
                gamma[[doc, topic]] = alpha[topic] + self.n_words_per_doc[doc] as f64 / t as f64;
            }
        }
        self.gamma = Some(gamma);
    }

    /// Train the LDA model on a corpus of tokenized documents.
    pub fn train(&mut self, corpus: Vec<Vec<usize>>, verbose: bool, max_iter: usize, tol: f64) {
        self.n_docs = corpus.len();
        self.vocab_size = corpus.iter().flat_map(|doc| doc.iter()).copied().max().map(|m| m + 1).unwrap_or(0);
        self.n_words_per_doc = corpus.iter().map(|doc| doc.len()).collect();
        self.corpus = corpus;

        self.initialize_parameters();
        let mut vlb = f64::NEG_INFINITY;

        for i in 0..max_iter {
            let old_vlb = vlb;

            self.e_step();
            self.m_step();

            vlb = self.vlb();
            let delta = vlb - old_vlb;

            if verbose {
                println!("Iteration {}: {:.3} (delta: {:.2})", i + 1, vlb, delta);
            }

            if delta < tol {
                break;
            }
        }
    }
}

fn dg(gamma: &Array2<f64>, d: usize, t: usize) -> f64 {
    gamma[[d, t]].digamma() - gamma.row(d).sum().digamma()
}

/// Smoothed Latent Dirichlet Allocation trained with collapsed Gibbs sampling.
pub struct SmoothedLDA {
    pub n_topics: usize,
    pub alpha: f64,
    pub beta: f64,
    pub phi: Option<Array2<f64>>,   // (V, T) word-topic distribution
    pub theta: Option<Array2<f64>>, // (D, T) doc-topic distribution
}

impl SmoothedLDA {
    pub fn new(n_topics: usize, alpha: Option<f64>, beta: Option<f64>) -> Self {
        let alpha = alpha.unwrap_or(50.0 / n_topics as f64);
        let beta = beta.unwrap_or(0.01);
        Self {
            n_topics,
            alpha,
            beta,
            phi: None,
            theta: None,
        }
    }

    /// Train the model on tokenized documents using collapsed Gibbs sampling.
    pub fn train(&mut self, corpus: &[Vec<usize>], n_gibbs: usize) {
        use rand::distributions::WeightedIndex;

        let d_count = corpus.len();
        let v_count = corpus
            .iter()
            .flat_map(|doc| doc.iter())
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let t = self.n_topics;
        let v = v_count;
        let d = d_count;
        let alpha = self.alpha;
        let beta = self.beta;

        // Flatten words and track their doc ids
        let mut words = Vec::new();
        let mut word_doc = Vec::new();
        for (doc_idx, doc) in corpus.iter().enumerate() {
            for &w in doc {
                words.push(w);
                word_doc.push(doc_idx);
            }
        }
        let total_words = words.len();
        if total_words == 0 || v == 0 {
            return;
        }

        let mut rng = rand::thread_rng();

        let mut c_wt = Array2::<usize>::zeros((v, t));
        let mut c_dt = Array2::<usize>::zeros((d, t));
        let mut sum_c_wt = vec![0usize; t];
        let mut sum_c_dt = vec![0usize; d];

        let mut assignments = Vec::with_capacity(total_words);
        for i in 0..total_words {
            let topic = rng.r#gen_range(0..t);
            assignments.push(topic);

            let w = words[i];
            let doc = word_doc[i];
            c_wt[[w, topic]] += 1;
            c_dt[[doc, topic]] += 1;
            sum_c_wt[topic] += 1;
            sum_c_dt[doc] += 1;
        }

        // Gibbs sampling
        for _ in 0..n_gibbs {
            for i in 0..total_words {
                let w = words[i];
                let doc = word_doc[i];
                let cur_topic = assignments[i];

                c_wt[[w, cur_topic]] -= 1;
                c_dt[[doc, cur_topic]] -= 1;
                sum_c_wt[cur_topic] -= 1;
                sum_c_dt[doc] -= 1;

                let mut p_vec = Vec::with_capacity(t);
                for topic in 0..t {
                    let p_w = (c_wt[[w, topic]] as f64 + beta) / (sum_c_wt[topic] as f64 + v as f64 * beta);
                    let p_t = (c_dt[[doc, topic]] as f64 + alpha) / (sum_c_dt[doc] as f64 + t as f64 * alpha);
                    p_vec.push(p_w * p_t);
                }

                let dist = WeightedIndex::new(&p_vec).unwrap();
                let new_topic = dist.sample(&mut rng);

                assignments[i] = new_topic;
                c_wt[[w, new_topic]] += 1;
                c_dt[[doc, new_topic]] += 1;
                sum_c_wt[new_topic] += 1;
                sum_c_dt[doc] += 1;
            }
        }

        // Compute phi and theta
        let mut phi = Array2::<f64>::zeros((v, t));
        for w in 0..v {
            for topic in 0..t {
                phi[[w, topic]] = (c_wt[[w, topic]] as f64 + beta) / (sum_c_wt[topic] as f64 + v as f64 * beta);
            }
        }

        let mut theta = Array2::<f64>::zeros((d, t));
        for doc in 0..d {
            for topic in 0..t {
                theta[[doc, topic]] = (c_dt[[doc, topic]] as f64 + alpha) / (sum_c_dt[doc] as f64 + t as f64 * alpha);
            }
        }

        self.phi = Some(phi);
        self.theta = Some(theta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothed_lda() {
        let corpus = vec![
            vec![0, 1, 2, 0, 1],
            vec![0, 1, 0, 2],
            vec![3, 4, 5, 3],
            vec![4, 5, 4, 3, 5],
        ];

        let mut model = SmoothedLDA::new(2, Some(1.0), Some(0.1));
        model.train(&corpus, 50);

        let phi = model.phi.as_ref().unwrap();
        assert_eq!(phi.nrows(), 6);
        assert_eq!(phi.ncols(), 2);

        let theta = model.theta.as_ref().unwrap();
        assert_eq!(theta.nrows(), 4);
        assert_eq!(theta.ncols(), 2);

        for doc in 0..4 {
            let sum: f64 = theta.row(doc).iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}

