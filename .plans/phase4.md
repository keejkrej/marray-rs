# Phase 4: numpy_ml Completion

Fill remaining gaps in the numpy_ml module.

## neural_nets.layers -- missing layer types

- [x] `Conv1D` -- 1D convolutional layer (forward + backward)
- [x] `Conv2D` -- 2D convolutional layer (forward + backward)
- [x] `MaxPool1D` / `AvgPool1D` -- 1D pooling layers
- [x] `MaxPool2D` / `AvgPool2D` -- 2D pooling layers
- [x] `BatchNorm` -- batch normalization layer
- [x] `Dropout` -- dropout regularization layer
- [x] `GRU` -- gated recurrent unit cell + layer
- [x] `Attention` -- scaled dot-product attention
- [x] `MultiHeadAttention` -- multi-headed attention
- [x] `Embedding` -- embedding layer

## neural_nets.models -- complete models

- [x] `VAE` -- variational autoencoder
- [x] `Word2Vec` -- word2vec (CBOW / skip-gram)

## trees -- decision tree module

- [x] `DecisionTree` -- numpy_ml decision tree (CART)
- [x] `GradientBoostedDecisionTree` -- gradient boosted trees
- [x] `RandomForest` -- random forest ensemble
- [x] Loss functions for trees (MSE, cross-entropy)

## ngram -- additional smoothing

- [x] `KneserNeyNGram` -- Kneser-Ney smoothing

## hmm -- additional HMM types

- [x] `GaussianHMM` -- HMM with Gaussian emissions

## bandits -- additional policies

- [x] `SoftmaxPolicy` -- softmax/boltzmann exploration

## rl_models -- additional agents

- [x] `SARSAAgent` -- SARSA agent

## utils.kernels -- additional kernels

- [x] `MaternKernel` -- Matérn kernel
- [x] `RationalQuadraticKernel` -- rational quadratic kernel

## linear_models -- additional models

- [x] `RidgeRegression` -- numpy_ml ridge regression
- [x] `LogisticRegression` -- numpy_ml logistic regression

## lda -- additional LDA variants

- [x] `SmoothedLDA` -- smoothed LDA
