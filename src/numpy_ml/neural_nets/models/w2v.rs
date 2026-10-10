use ndarray::{Array1, Array2};
use rand::Rng;

/// Word2Vec model supporting both Continuous Bag of Words (CBOW)
/// and Skip-gram architectures with Noise Contrastive Estimation / Negative Sampling.
pub struct Word2Vec {
    /// Number of words in the vocabulary
    pub vocab_size: usize,
    /// Dimension of word embedding vectors
    pub embedding_dim: usize,
    /// Context window size (radius to left and right)
    pub context_len: usize,
    /// Whether to use skip-gram (true) or CBOW (false)
    pub skip_gram: bool,
    /// Number of negative samples per positive target
    pub num_negative_samples: usize,
    /// Exponent applied to unigram counts for negative sampling (default 0.75)
    pub noise_dist_power: f64,
    /// Input word embeddings: shape (vocab_size, embedding_dim)
    pub embeddings: Array2<f64>,
    /// Context / output weights: shape (vocab_size, embedding_dim)
    pub context_weights: Array2<f64>,
    /// Cumulative distribution or unigram probabilities for negative sampling
    pub noise_probs: Vec<f64>,
}

impl Word2Vec {
    /// Create a new Word2Vec model.
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        context_len: usize,
        skip_gram: bool,
        num_negative_samples: usize,
    ) -> Self {
        let gain = 1.0;
        let bound = (gain / embedding_dim as f64).sqrt();
        let mut rng = rand::thread_rng();

        let embeddings = Array2::from_shape_fn((vocab_size, embedding_dim), |_| {
            rng.gen_range(-bound..bound)
        });
        let context_weights = Array2::from_shape_fn((vocab_size, embedding_dim), |_| {
            rng.gen_range(-bound..bound)
        });

        // Default uniform noise distribution
        let uniform = 1.0 / vocab_size as f64;
        let noise_probs = vec![uniform; vocab_size];

        Self {
            vocab_size,
            embedding_dim,
            context_len,
            skip_gram,
            num_negative_samples,
            noise_dist_power: 0.75,
            embeddings,
            context_weights,
            noise_probs,
        }
    }

    /// Set noise distribution from corpus token frequencies.
    pub fn set_token_counts(&mut self, counts: &[usize]) {
        assert_eq!(counts.len(), self.vocab_size);
        let mut probs: Vec<f64> = counts
            .iter()
            .map(|&c| (c as f64).powf(self.noise_dist_power))
            .collect();
        let total: f64 = probs.iter().sum();
        if total > 0.0 {
            for p in probs.iter_mut() {
                *p /= total;
            }
        }
        self.noise_probs = probs;
    }

    /// Sample negative token indices from the noise distribution.
    pub fn sample_negative(&self, exclude: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(self.num_negative_samples);

        while samples.len() < self.num_negative_samples {
            // Roulette-wheel / discrete sampling
            let r: f64 = rng.r#gen();
            let mut acc = 0.0;
            let mut chosen = 0;
            for (idx, &p) in self.noise_probs.iter().enumerate() {
                acc += p;
                if r <= acc {
                    chosen = idx;
                    break;
                }
            }
            if chosen != exclude {
                samples.push(chosen);
            } else if self.vocab_size <= 1 {
                samples.push(chosen);
            }
        }
        samples
    }

    #[inline]
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x.clamp(-15.0, 15.0)).exp())
    }

    /// Perform a single CBOW training step on a context window and target word.
    ///
    /// Context words are averaged to predict the target word.
    pub fn train_step_cbow(&mut self, context: &[usize], target: usize, lr: f64) -> f64 {
        if context.is_empty() {
            return 0.0;
        }

        // 1. Average context embeddings
        let mut h = Array1::<f64>::zeros(self.embedding_dim);
        for &c in context {
            for d in 0..self.embedding_dim {
                h[d] += self.embeddings[[c, d]];
            }
        }
        h /= context.len() as f64;

        // 2. Negative samples
        let neg_samples = self.sample_negative(target);

        // 3. Positive target gradient
        let dot_pos: f64 = (0..self.embedding_dim)
            .map(|d| self.context_weights[[target, d]] * h[d])
            .sum();
        let sig_pos = Self::sigmoid(dot_pos);
        let pos_loss = -sig_pos.max(1e-12).ln();
        let g_pos = sig_pos - 1.0;

        let mut dh = Array1::<f64>::zeros(self.embedding_dim);
        for d in 0..self.embedding_dim {
            dh[d] += g_pos * self.context_weights[[target, d]];
            self.context_weights[[target, d]] -= lr * g_pos * h[d];
        }

        // 4. Negative samples gradient
        let mut neg_loss = 0.0;
        for &neg in &neg_samples {
            let dot_neg: f64 = (0..self.embedding_dim)
                .map(|d| self.context_weights[[neg, d]] * h[d])
                .sum();
            let sig_neg = Self::sigmoid(dot_neg);
            neg_loss += -(1.0 - sig_neg).max(1e-12).ln();
            let g_neg = sig_neg;

            for d in 0..self.embedding_dim {
                dh[d] += g_neg * self.context_weights[[neg, d]];
                self.context_weights[[neg, d]] -= lr * g_neg * h[d];
            }
        }

        // 5. Backpropagate to context embeddings
        let ctx_scale = lr / context.len() as f64;
        for &c in context {
            for d in 0..self.embedding_dim {
                self.embeddings[[c, d]] -= ctx_scale * dh[d];
            }
        }

        pos_loss + neg_loss
    }

    /// Perform a single Skip-gram training step on a target word and its context words.
    ///
    /// Target word is used to predict each context word.
    pub fn train_step_skipgram(&mut self, target: usize, context: &[usize], lr: f64) -> f64 {
        if context.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut d_target = Array1::<f64>::zeros(self.embedding_dim);

        for &ctx in context {
            let neg_samples = self.sample_negative(ctx);

            // Positive context prediction
            let dot_pos: f64 = (0..self.embedding_dim)
                .map(|d| self.context_weights[[ctx, d]] * self.embeddings[[target, d]])
                .sum();
            let sig_pos = Self::sigmoid(dot_pos);
            total_loss += -sig_pos.max(1e-12).ln();
            let g_pos = sig_pos - 1.0;

            for d in 0..self.embedding_dim {
                d_target[d] += g_pos * self.context_weights[[ctx, d]];
                self.context_weights[[ctx, d]] -= lr * g_pos * self.embeddings[[target, d]];
            }

            // Negative samples prediction
            for &neg in &neg_samples {
                let dot_neg: f64 = (0..self.embedding_dim)
                    .map(|d| self.context_weights[[neg, d]] * self.embeddings[[target, d]])
                    .sum();
                let sig_neg = Self::sigmoid(dot_neg);
                total_loss += -(1.0 - sig_neg).max(1e-12).ln();
                let g_neg = sig_neg;

                for d in 0..self.embedding_dim {
                    d_target[d] += g_neg * self.context_weights[[neg, d]];
                    self.context_weights[[neg, d]] -= lr * g_neg * self.embeddings[[target, d]];
                }
            }
        }

        // Update target embedding
        for d in 0..self.embedding_dim {
            self.embeddings[[target, d]] -= lr * d_target[d];
        }

        total_loss
    }

    /// Train on a sequence of token IDs.
    pub fn fit_sequence(&mut self, tokens: &[usize], n_epochs: usize, lr: f64) -> Vec<f64> {
        let n = tokens.len();
        let mut epoch_losses = Vec::with_capacity(n_epochs);

        for _ in 0..n_epochs {
            let mut epoch_loss = 0.0;
            let mut steps = 0;

            for i in 0..n {
                let target = tokens[i];
                let mut context = Vec::new();

                let start = i.saturating_sub(self.context_len);
                let end = (i + self.context_len + 1).min(n);

                for j in start..end {
                    if j != i {
                        context.push(tokens[j]);
                    }
                }

                if !context.is_empty() {
                    let step_loss = if self.skip_gram {
                        self.train_step_skipgram(target, &context, lr)
                    } else {
                        self.train_step_cbow(&context, target, lr)
                    };
                    epoch_loss += step_loss;
                    steps += 1;
                }
            }

            epoch_losses.push(epoch_loss / steps.max(1) as f64);
        }

        epoch_losses
    }

    /// Get the embedding vector for a given word ID.
    pub fn get_embedding(&self, word_id: usize) -> Array1<f64> {
        self.embeddings.slice(ndarray::s![word_id, ..]).to_owned()
    }

    /// Find the top-k most similar words using cosine similarity.
    pub fn most_similar(&self, word_id: usize, top_k: usize) -> Vec<(usize, f64)> {
        let query = self.get_embedding(word_id);
        let query_norm = query.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);

        let mut similarities = Vec::with_capacity(self.vocab_size);
        for w in 0..self.vocab_size {
            if w == word_id {
                continue;
            }
            let emb = self.embeddings.slice(ndarray::s![w, ..]);
            let emb_norm = emb.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            let dot: f64 = (0..self.embedding_dim).map(|d| query[d] * emb[d]).sum();
            let cos_sim = dot / (query_norm * emb_norm);
            similarities.push((w, cos_sim));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }
}
