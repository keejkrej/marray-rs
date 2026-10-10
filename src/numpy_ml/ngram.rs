use std::collections::HashMap;

/// Base N-gram language model operating on integer token IDs.
pub struct NGramBase {
    pub n: usize,
    pub counts: Vec<HashMap<Vec<usize>, usize>>,
    pub n_words: Vec<usize>,
    pub n_tokens: Vec<usize>,
}

impl NGramBase {
    /// Train the n-gram model on a corpus of token sequences.
    pub fn train(&mut self, corpus: &[Vec<usize>]) {
        let mut grams: Vec<Vec<Vec<usize>>> = (1..=self.n).map(|_| Vec::new()).collect();
        let mut total_words = 0;
        let mut tokens = HashMap::new();
        tokens.insert(usize::MAX, 1); // placeholder for unknown token handling

        for words in corpus {
            total_words += words.len();
            for &w in words {
                tokens.insert(w, tokens.get(&w).unwrap_or(&0) + 1);
            }

            for gram_size in 1..=self.n {
                let padded = Self::pad(words, gram_size);
                for window in padded.windows(gram_size) {
                    grams[gram_size - 1].push(window.to_vec());
                }
            }
        }

        self.counts = (0..self.n).map(|i| {
            let mut counts = HashMap::new();
            for gram in &grams[i] {
                *counts.entry(gram.clone()).or_insert(0) += 1;
            }
            counts
        }).collect();

        self.n_words = vec![0; self.n + 1];
        self.n_words[1] = total_words;
        for gram_size in 2..=self.n {
            self.n_words[gram_size] = self.counts[gram_size - 1].values().sum();
        }

        self.n_tokens = vec![0; self.n + 1];
        self.n_tokens[1] = tokens.len();
        for gram_size in 2..=self.n {
            self.n_tokens[gram_size] = self.counts[gram_size - 1].len();
        }
    }

    fn pad(words: &[usize], gram_size: usize) -> Vec<usize> {
        let pad_len = (gram_size - 1).max(1);
        let bol = vec![usize::MAX; pad_len];
        let eol = vec![usize::MAX - 1; pad_len];
        let mut result = Vec::with_capacity(pad_len * 2 + words.len());
        result.extend(&bol);
        result.extend(words);
        result.extend(&eol);
        result
    }

    /// Compute the log probability of a token sequence under the `n`-gram model.
    pub fn log_prob(&self, words: &[usize], n: usize, log_ngram_prob: &dyn Fn(&[usize], &Self) -> f64) -> Result<f64, String> {
        if n > self.n {
            return Err(format!("Model only trained up to {}-grams", self.n));
        }
        if n > words.len() {
            return Err("Not enough words for the requested n-gram size".to_string());
        }

        let padded = Self::pad(words, n);
        Ok(padded.windows(n).map(|gram| log_ngram_prob(gram, self)).sum())
    }

    /// Compute cross-entropy of a sequence under the model.
    pub fn cross_entropy(&self, words: &[usize], n: usize, log_ngram_prob: &dyn Fn(&[usize], &Self) -> f64) -> Result<f64, String> {
        let n_ngrams = words.len().saturating_sub(n - 1).max(1);
        Ok(-self.log_prob(words, n, log_ngram_prob)? / n_ngrams as f64)
    }

    /// Compute perplexity of a sequence under the model.
    pub fn perplexity(&self, words: &[usize], n: usize, log_ngram_prob: &dyn Fn(&[usize], &Self) -> f64) -> Result<f64, String> {
        Ok(self.cross_entropy(words, n, log_ngram_prob)?.exp())
    }
}

/// Maximum-likelihood (unsmoothed) N-gram model.
pub struct MLENGram {
    base: NGramBase,
}

impl MLENGram {
    /// Create a new MLE N-gram model.
    pub fn new(n: usize) -> Self {
        Self {
            base: NGramBase { n, counts: Vec::new(), n_words: Vec::new(), n_tokens: Vec::new() },
        }
    }

    /// Train the model.
    pub fn train(&mut self, corpus: &[Vec<usize>]) {
        self.base.train(corpus);
    }

    /// Log probability.
    pub fn log_prob(&self, words: &[usize], n: usize) -> Result<f64, String> {
        self.base.log_prob(words, n, &|gram, base| {
            let gram_size = gram.len();
            let count = *base.counts[gram_size - 1].get(gram).unwrap_or(&0) as f64;
            let denom = if gram_size > 1 {
                *base.counts[gram_size - 2].get(&gram[..gram_size - 1].to_vec()).unwrap_or(&0) as f64
            } else {
                base.n_words[1] as f64
            };
            if count > 0.0 && denom > 0.0 { count.ln() - denom.ln() } else { f64::NEG_INFINITY }
        })
    }

    /// Perplexity.
    pub fn perplexity(&self, words: &[usize], n: usize) -> Result<f64, String> {
        self.base.perplexity(words, n, &|gram, base| {
            let gram_size = gram.len();
            let count = *base.counts[gram_size - 1].get(gram).unwrap_or(&0) as f64;
            let denom = if gram_size > 1 {
                *base.counts[gram_size - 2].get(&gram[..gram_size - 1].to_vec()).unwrap_or(&0) as f64
            } else {
                base.n_words[1] as f64
            };
            if count > 0.0 && denom > 0.0 { count.ln() - denom.ln() } else { f64::NEG_INFINITY }
        })
    }
}

/// Additive (Lidstone) smoothed N-gram model.
pub struct AdditiveNGram {
    base: NGramBase,
    k: f64,
}

impl AdditiveNGram {
    /// Create a new additive-smoothed N-gram model with pseudocount `k`.
    pub fn new(n: usize, k: f64) -> Self {
        Self {
            base: NGramBase { n, counts: Vec::new(), n_words: Vec::new(), n_tokens: Vec::new() },
            k,
        }
    }

    /// Train the model.
    pub fn train(&mut self, corpus: &[Vec<usize>]) {
        self.base.train(corpus);
    }

    /// Log probability.
    pub fn log_prob(&self, words: &[usize], n: usize) -> Result<f64, String> {
        self.base.log_prob(words, n, &|gram, base| {
            let gram_size = gram.len();
            let count = *base.counts[gram_size - 1].get(gram).unwrap_or(&0) as f64;
            let denom = if gram_size > 1 {
                *base.counts[gram_size - 2].get(&gram[..gram_size - 1].to_vec()).unwrap_or(&0) as f64
            } else {
                base.n_words[1] as f64
            };
            let v = base.n_tokens[gram_size] as f64;
            ((count + self.k) / (denom + self.k * v)).ln()
        })
    }

    /// Perplexity.
    pub fn perplexity(&self, words: &[usize], n: usize) -> Result<f64, String> {
        self.base.perplexity(words, n, &|gram, base| {
            let gram_size = gram.len();
            let count = *base.counts[gram_size - 1].get(gram).unwrap_or(&0) as f64;
            let denom = if gram_size > 1 {
                *base.counts[gram_size - 2].get(&gram[..gram_size - 1].to_vec()).unwrap_or(&0) as f64
            } else {
                base.n_words[1] as f64
            };
            let v = base.n_tokens[gram_size] as f64;
            ((count + self.k) / (denom + self.k * v)).ln()
        })
    }
}

/// Interpolated Kneser-Ney smoothed N-gram model.
pub struct KneserNeyNGram {
    base: NGramBase,
    d: f64,
}

impl KneserNeyNGram {
    /// Create a new Kneser-Ney smoothed N-gram model with discount `d` (default 0.75).
    pub fn new(n: usize, d: f64) -> Self {
        Self {
            base: NGramBase {
                n,
                counts: Vec::new(),
                n_words: Vec::new(),
                n_tokens: Vec::new(),
            },
            d: d.clamp(0.01, 0.99),
        }
    }

    /// Train the model.
    pub fn train(&mut self, corpus: &[Vec<usize>]) {
        self.base.train(corpus);
    }

    fn kn_prob(&self, gram: &[usize]) -> f64 {
        let k = gram.len();
        if k <= 1 {
            let w = gram[0];
            if self.base.counts.len() >= 2 {
                let bigrams = &self.base.counts[1];
                let mut continuation_count = 0;
                let mut total_continuation = 0;
                for (bg, &cnt) in bigrams.iter() {
                    if cnt > 0 {
                        total_continuation += 1;
                        if bg[1] == w {
                            continuation_count += 1;
                        }
                    }
                }
                if total_continuation > 0 {
                    return (continuation_count as f64).max(0.1) / total_continuation as f64;
                }
            }
            let count = if !self.base.counts.is_empty() {
                *self.base.counts[0].get(gram).unwrap_or(&0) as f64
            } else {
                0.0
            };
            let total = self.base.n_words.get(1).copied().unwrap_or(1) as f64;
            (count + 0.1) / (total + 1.0)
        } else {
            let context = &gram[..k - 1];
            let c_cw = *self.base.counts[k - 1].get(gram).unwrap_or(&0) as f64;
            let c_c = *self.base.counts[k - 2].get(&context.to_vec()).unwrap_or(&0) as f64;

            let p_lower = self.kn_prob(&gram[1..]);

            if c_c > 0.0 {
                let unique_completions = self.base.counts[k - 1]
                    .keys()
                    .filter(|key| key.starts_with(context))
                    .count() as f64;

                let lambda = (self.d / c_c) * unique_completions;
                let direct = (c_cw - self.d).max(0.0) / c_c;
                direct + lambda * p_lower
            } else {
                p_lower
            }
        }
    }

    /// Log probability.
    pub fn log_prob(&self, words: &[usize], n: usize) -> Result<f64, String> {
        self.base.log_prob(words, n, &|gram, _base| {
            self.kn_prob(gram).max(1e-15).ln()
        })
    }

    /// Perplexity.
    pub fn perplexity(&self, words: &[usize], n: usize) -> Result<f64, String> {
        self.base.perplexity(words, n, &|gram, _base| {
            self.kn_prob(gram).max(1e-15).ln()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kneser_ney_ngram() {
        let corpus = vec![
            vec![0, 1, 2],
            vec![0, 1, 1],
            vec![0, 2, 1],
        ];

        let mut model = KneserNeyNGram::new(2, 0.75);
        model.train(&corpus);

        let pp = model.perplexity(&vec![0, 1, 2], 2).unwrap();
        assert!(pp.is_finite());
        assert!(pp > 0.0);
    }
}

