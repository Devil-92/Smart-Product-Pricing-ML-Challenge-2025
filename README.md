# ============================================================
# Project: Multimodal Price Prediction
# Challenge: ML Challenge 2025 – Smart Product Pricing
# ============================================================

project_name: "Multimodal Price Prediction"
challenge: "ML Challenge 2025 – Smart Product Pricing"

# -----------------------
# Project Overview
# -----------------------
overview: >
  This repository implements a multimodal regression system designed to predict
  product prices using catalog text, product descriptions, product images, and structured metadata.
  No external datasets or historical pricing information are used, following challenge constraints.

# -----------------------
# Objective
# -----------------------
objective: >
  Develop a regression model capable of predicting product prices using only text, images,
  and metadata. The system is evaluated using SMAPE and aims for competitive generalization
  across diverse product categories.

# -----------------------
# Dataset Description
# -----------------------
dataset:
  total_samples: 150000
  train_samples: 75000
  test_samples: 75000
  
  modalities:
    - text: "Product titles, descriptions, and catalog content"
    - images: "Product images obtained from provided URLs"
    - metadata: "Brand, quantity, category indicators"

# -----------------------
# Approach & Feature Engineering
# -----------------------
approach:

  text_features:
    - "TF-IDF vectors for high-dimensional lexical patterns"
    - "SVD-reduced components capturing latent semantic structure"
    - "SentenceTransformer embeddings for semantic representation"

  image_features:
    - "EfficientNet embeddings extracted from product images"
    - "CLIP vision embeddings for multimodal semantic alignment"
    - "PCA dimensionality reduction for computational efficiency"

  structured_features:
    - "Brand-based frequency and target-encoding signals"
    - "Quantity normalization and scaling"
    - "Keyword indicators derived from textual metadata"

  models:
    - "LightGBM"
    - "XGBoost"
    - "CatBoost"
    - "MLP networks for embedding fusion"

  ensemble_strategy:
    type: "stacking"
    description: >
      A meta-model integrates predictions from text-based models,
      image-based models, and tree ensemble models. This stacking approach
      improves robustness and reduces SMAPE by leveraging complementary signals.

# -----------------------
# Evaluation
# -----------------------
evaluation:
  metric: "SMAPE - Symmetric Mean Absolute Percentage Error"
  goal: "< 40 SMAPE"

# -----------------------
# Repository Contents
# -----------------------
repository_contents:
  includes:
    - "Text preprocessing scripts: tokenization, TF-IDF, SVD, embedding generation"
    - "Image preprocessing and embedding extraction pipelines"
    - "Model training workflows for all feature classes"
    - "Stacking ensemble training framework"
    - "Evaluation utilities for SMAPE, metrics, and validation"
    - "Inference pipeline for generating final challenge submissions"

# -----------------------
# Summary
# -----------------------
summary: >
  This repository provides a complete multimodal solution for product price prediction,
  integrating text, image, and metadata-based features. The final system employs
  a robust stacking ensemble to achieve consistent performance across varied product categories.
