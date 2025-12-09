project_name: "Multimodal Price Prediction"
challenge: "ML Challenge 2025 – Smart Product Pricing"

objective: >
  Develop a regression model capable of predicting product prices using only
  catalog text, product descriptions, and product images. No external datasets
  or historical pricing information are used.

dataset:
  total_samples: 150000
  train_samples: 75000
  test_samples: 75000
  modalities:
    - text: product titles, descriptions, catalog content
    - images: product images from provided URLs
    - metadata: brand, quantity, and category indicators

approach:
  text_features:
    - TF-IDF vectors
    - SVD-reduced components
    - SentenceTransformer embeddings
  image_features:
    - EfficientNet or CLIP embeddings
    - PCA dimensionality reduction
  structured_features:
    - brand signals
    - quantity normalization
    - keyword indicators
  models:
    - LightGBM
    - XGBoost
    - CatBoost
    - MLP for embedding fusion
  ensemble_strategy:
    type: "stacking"
    description: >
      A meta-model integrates predictions from all text-based, image-based, and
      gradient boosting models to improve generalization and reduce SMAPE.

evaluation:
  metric: "SMAPE - Symmetric Mean Absolute Percentage Error"
  goal: "< 40 SMAPE"

repository_contents:
  includes:
    - preprocessing scripts for text, images, and metadata
    - model training pipelines
    - ensemble/stacking framework
    - evaluation utilities
    - inference pipeline for submission generation

summary: >
  This repository provides a complete multimodal solution for price prediction
  using combined text, image, and structured features. The final system applies
  a stacking ensemble to achieve robust performance across diverse product types.
