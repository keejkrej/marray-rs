pub mod vae;
pub mod w2v;

pub use vae::{BernoulliVAE, VAE};
pub use w2v::Word2Vec;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_vae_forward_backward_sample() {
        let n_in = 8;
        let t_dim = 2;
        let latent_dim = 4;
        let mut vae = BernoulliVAE::new(n_in, t_dim, latent_dim);

        // Dummy data (batch size 4, 8 features with values in [0, 1])
        let x = Array2::from_shape_fn((4, n_in), |(i, j)| ((i + j) % 2) as f64 * 0.8 + 0.1);
        let recon = vae.forward(&x, true);
        assert_eq!(recon.dim(), (4, n_in));
        for &val in recon.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Backward
        let dx = vae.backward(&x, &recon);
        assert_eq!(dx.dim(), (4, n_in));

        // Update
        vae.update_params(None);

        // Sample
        let samples = vae.sample(5);
        assert_eq!(samples.dim(), (5, n_in));

        // Fit for 2 epochs
        let losses = vae.fit(&x, 2, 2);
        assert_eq!(losses.len(), 2);
    }

    #[test]
    fn test_word2vec_cbow_and_skipgram() {
        let vocab_size = 10;
        let embedding_dim = 8;
        let context_len = 2;

        // Test CBOW
        let mut w2v_cbow = Word2Vec::new(vocab_size, embedding_dim, context_len, false, 3);
        let tokens = vec![0, 1, 2, 3, 4, 1, 2, 5, 6, 7, 8, 9, 0, 1, 2];
        let losses_cbow = w2v_cbow.fit_sequence(&tokens, 3, 0.05);
        assert_eq!(losses_cbow.len(), 3);
        let emb0 = w2v_cbow.get_embedding(0);
        assert_eq!(emb0.len(), embedding_dim);

        let similar = w2v_cbow.most_similar(0, 3);
        assert_eq!(similar.len(), 3);

        // Test Skip-Gram
        let mut w2v_sg = Word2Vec::new(vocab_size, embedding_dim, context_len, true, 3);
        let counts = vec![2, 3, 3, 1, 1, 1, 1, 1, 1, 1];
        w2v_sg.set_token_counts(&counts);
        let losses_sg = w2v_sg.fit_sequence(&tokens, 3, 0.05);
        assert_eq!(losses_sg.len(), 3);
        let emb1 = w2v_sg.get_embedding(1);
        assert_eq!(emb1.len(), embedding_dim);
    }
}
