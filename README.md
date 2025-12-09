---
# Multimodal Price Prediction  
### ML Challenge 2025 – Smart Product Pricing
---

## Overview
This repository implements a multimodal regression system designed to predict product prices using catalog text, product descriptions, product images, and structured metadata.  
No external datasets or historical pricing information are used, following the challenge constraints.

---

## Objective
Develop a regression model capable of predicting product prices using only text, images, and metadata.  
The system is evaluated using **SMAPE** and aims for competitive generalization across diverse product categories.

---

## Dataset

- **Total samples:** 150,000  
- **Training samples:** 75,000  
- **Test samples:** 75,000  

### Modalities
- **Text:** Product titles, descriptions, catalog content  
- **Images:** Product images obtained from provided URLs  
- **Metadata:** Brand, quantity, category indicators  

---

## Approach

### Text Features
- TF-IDF vectors for high-dimensional lexical patterns  
- SVD-reduced latent semantic components  
- SentenceTransformer semantic embeddings  

### Image Features
- EfficientNet embeddings extracted from product images  
- CLIP vision embeddings for multimodal semantic alignment  
- PCA for dimensionality reduction  

### Structured Features
- Brand frequency and encoding signals  
- Quantity normalization and scaling  
- Keyword indicators derived from metadata  

### Models
- LightGBM  
- XGBoost  
- CatBoost  
- MLP networks for embedding fusion  

### Ensemble Strategy
**Type:** Stacking  
**Description:**  
A meta-model integrates predictions from text-based models, image-based models, and gradient boosting models.  
This stacking approach improves robustness and reduces SMAPE by leveraging complementary multimodal signals.

---

## Evaluation
- **Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)  
- **Goal:** < 40 SMAPE  

---

## Repository Contents

Includes:

- Text preprocessing scripts: tokenization, TF-IDF, SVD, embedding generation  
- Image preprocessing and embedding extraction pipelines  
- Model training workflows for all modalities  
- Stacking ensemble training framework  
- Evaluation utilities for SMAPE and validation  
- Inference pipeline for generating final challenge submissions  

---

## Summary
This repository provides an end-to-end multimodal solution for product price prediction.  
By combining text, image, and structured metadata features, and applying a robust stacking ensemble, the system achieves consistent performance across varied product types.

---

