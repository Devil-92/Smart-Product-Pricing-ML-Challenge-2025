# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** [TedheMedhe123]  
**Team Members:** [Ritesh Kumar, Arpit Singh, MAMUN CHOWDHURY]
**Submission Date:** [12/10/2025]

---

## 1. Executive Summary
Our solution predicts product prices using a multimodal LightGBM model that integrates engineered features from text, structured data, and image embeddings. The approach centered on detailed text parsing to extract key attributes like brand and pack size, leveraging a pre-trained ResNet50 for image feature extraction, and maximizing final performance through automated hyperparameter tuning.



---

## 2. Methodology Overview

### 2.1 Problem Analysis

We interpreted the challenge as a multimodal regression task, where the primary goal was to accurately predict a product's price by leveraging a combination of unstructured text and visual data. The strict prohibition of external price lookups meant that success depended entirely on our ability to engineer high-quality features from the provided catalog_content and image_link. Our initial exploratory data analysis (EDA) was crucial and uncovered several key insights that shaped our entire strategy.

### Key Observations:

Log-Normal Price Distribution: We observed that the price target variable was heavily right-skewed, with a long tail of high-priced items. Applying a log transformation (np.log1p) was essential to normalize the distribution, making it more suitable for model training and reducing the disproportionate influence of outliers.

Structured Text Content: The catalog_content was not just a block of text but a semi-structured field containing distinct, parsable sections like "Item Name," "Bullet Points," "Product Description," and explicit "Value" and "Unit" information. This insight drove our decision to parse these fields into separate, structured features rather than treating the entire text as a single entity.

Implicit Quantitative Data: Beyond the explicit Value and Unit, the text frequently contained crucial quantitative information, such as "Pack of 12" or "16 ct." Extracting this Item Pack Quantity (IPQ) was identified as a high-priority feature engineering task, as price is often directly correlated with quantity.



### 2.2 Solution Strategy
Our high-level approach was multimodal learning with an early fusion strategy. We aimed to create a single, comprehensive feature vector for each product by combining information from all available data sources (structured, text, and image) and then training one powerful model on this unified representation.

Approach Type: Single Model (LightGBM) with Multimodal Features
Core Innovation: The core innovation of our solution was the development of a robust and holistic feature engineering pipeline. Instead of relying on a complex new algorithm, we focused on meticulously transforming the raw data by integrating parsed structured attributes (like brand, pack size, and unit), NLP-derived features (TF-IDF), and deep learning-based image embeddings (from ResNet50) into a single, high-dimensional matrix that a gradient boosting model could effectively exploit.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Our model is an early fusion multimodal architecture. The core design philosophy is to transform all diverse data sources (structured text, natural language, and images) into a single, unified numerical feature vector before feeding it into a powerful gradient boosting model. The complexity lies not in a deep neural network, but in the sophisticated pre-processing and feature engineering pipeline that creates this comprehensive representation for each product.

This approach allows a single, highly optimized LightGBM model to find complex patterns and interactions between the different data modalities simultaneously.

A simplified flowchart of the architecture is as follows:
graph TD

    A[Input: catalog_content] --> B{Text Parsing};
    A --> C{TF-IDF Vectorizer};
    D[Input: image_link] --> E{Image Download};

    B --> F[Structured Features <br/> (Brand, Pack Size, Value, Unit)];
    F --> G[Preprocessing <br/> (Scaling & One-Hot Encoding)];

    C --> H[NLP Features <br/> (5000-dim TF-IDF Vector)];

    E --> I{Pre-trained ResNet50};
    I --> J[Image Features <br/> (2048-dim Embedding Vector)];

    G --> K([Feature Concatenation <br/> hstack]);
    H --> K;
    J --> K;

    K --> L[Combined Feature Matrix <br/> (16,838 Features)];
    L --> M{Optimized LightGBM Regressor};
    M --> N[Output: Predicted log_price];
    N --> O{Exponentiation (np.expm1)};
    O --> P[Final Predicted Price];

### 3.2 Model Components

Text Processing Pipeline:

Preprocessing steps: Text from item_name, bullet_points, and description was concatenated into a single field. This combined text was then tokenized, converted to lowercase, and stripped of common English stop words.

Model type: TfidfVectorizer (Term Frequency-Inverse Document Frequency) from scikit-learn.

Key parameters:

   max_features: 5000 (to keep the vocabulary to the most relevant terms).

   ngram_range: (1, 2) (to capture both single words and two-word phrases). 

   stop_words: 'english'.

**Image Processing Pipeline:**

Preprocessing steps: Each image was resized to a uniform 224x224 pixel dimension. The pixel values were then normalized according to the specific requirements of the ResNet50 model using its dedicated preprocess_input function.

Model type: ResNet50, a pre-trained Convolutional Neural Network (CNN) used as a feature extractor.

Key parameters:

weights: 'imagenet' (to load the weights learned from the ImageNet dataset).

include_top: False (to remove the final classification layer).

pooling: 'avg' (to apply global average pooling and get a single 2048-dimensional feature vector per image).


---


## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 54.07%


## 5. Conclusion
Our solution successfully predicted product prices by fusing structured, text, and image features into a single, powerful LightGBM model. This multimodal approach achieved a final validation SMAPE score of 54.07%, a significant improvement over the text-only baseline, demonstrating that a holistic feature engineering strategy is paramount. The key lesson learned is that combining data modalities unlocks predictive power that no single source can provide on its own.

---
