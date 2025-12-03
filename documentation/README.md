 🗺️ SMART PRODUCT PRICING — PERFORMANCE-OPTIMIZED ROADMAP
Target: SMAPE < 40


Phase 1: Data Understanding & Exploration (Days 1–3)
🔍 1.1 Dataset Profiling
Load train.csv and test.csv.
Inspect data types, missing values, distributions.
Log-transform target to stabilize regression:
 
df['log_price'] = np.log1p(df['price'])

✅ Reduces variance and improves regression stability.

🧠 1.2 Text Structure Analysis
Parse catalog_content into title, description, bullet points.
Extract metrics:

Text length (chars, words)
Unique token counts
Missing/short entries
Identify keywords for brand and quantity (ml, pack, count).
Correlate features with log-price.
🖼️ 1.3 Image Exploration
Download sample images (50–100).
Extract:

Resolution, brightness, entropy, aspect ratio
Missing/invalid images
Flag low-quality images.

Phase 2: Feature Engineering (Days 4–7)
🧾 2.1 Text Feature Extraction
Structured Features

Standardize:

Brand (brand_name)
Quantity (normalize to grams/ml/count)
Category from title
Binary flags: has_brand, has_quantity, has_premium_word
Linguistic Features

len_title, len_desc, title_desc_ratio
flesch_reading_ease, avg_word_len
TF-IDF → SVD (100D)
Sentence embeddings (MiniLM or sentence-transformers)
✅ High-text features: 5–10 point SMAPE gain.

🖼️ 2.2 Image Feature Extraction
Extract embeddings using EfficientNet or CLIP ViT:
 
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)

Normalize and reduce with PCA (64–128D)
Compute CLIP similarity: cosine(text_emb, image_emb)
Compute image stats: image_entropy, contrast, dominant_color
Flags: has_image, image_quality_score
✅ Adds 5–8 points SMAPE improvement.

🧩 2.3 Combined / Cross-modal Features
CLIP similarity score for text-image coherence.
Elementwise fusion:
 
fusion_features = [text_emb - img_emb, text_emb * img_emb]

Concatenate structured + text + image features.

Phase 3: Model Development (Days 8–14)
⚙️ 3.1 Baseline Models
Train LightGBM on structured + text features.
Use log-price as target.
Evaluate via 5-fold GroupKFold (brand/category).
✅ SMAPE ~55–58

🤖 3.2 Model Variants
Model
Input
Purpose
XGBoost
structured + text
Strong non-linear baseline
CatBoost
categorical
Brand/category handling
Ridge / Lasso
TF-IDF
Linear baseline
MLP
embeddings
Nonlinear fusion
ViT / EfficientNet
image
Vision-only baseline

🧬 3.3 Fusion Models
Early Fusion: concat features → MLP
Intermediate Fusion: cross-attention transformer (text + image)
Late Fusion: weighted ensemble of text + image + fusion models
✅ Fusion drops SMAPE to ~45–48

Phase 4: Model Optimization (Days 15–18)
🎛️ 4.1 Hyperparameter Tuning (Optuna)
 
params = {
'num_leaves': trial.suggest_int(20, 150),
'learning_rate': trial.suggest_float(0.005, 0.05, log=True),
'feature_fraction': trial.suggest_float(0.5, 0.9),
'bagging_fraction': trial.suggest_float(0.5, 0.9),
'lambda_l1': trial.suggest_float(0, 10),
'lambda_l2': trial.suggest_float(0, 10),
'objective': 'regression_l1'
}

✅ 3–4 point SMAPE reduction.

📊 4.2 Cross-validation Strategy
Use GroupKFold by brand/category to reduce leakage.
Track fold-level SMAPE variance (<2%).
🧠 4.3 Regularization & Loss
MAE/Huber objective
Dropout + feature subsampling (feature_fraction=0.7)
Multi-seed averaging
✅ Smooths predictions → 1–2 points lower SMAPE.

Phase 5: Ensembling & Stacking (Days 19–20)
🧩 5.1 Weighted Blending
 
final_pred = (
0.4 * lgb_pred +
0.3 * xgb_pred +
0.2 * cat_pred +
0.1 * fusion_pred
)

🧠 5.2 Meta-Model (Stacking)
Generate OOF predictions for all models
Train LightGBM meta-learner on OOFs
⚖️ 5.3 Dynamic Ensemble by Price
Price Range
Weight Strategy
Low (<200)
0.7 text + 0.3 fusion
Medium (200–1000)
0.5 text + 0.5 fusion
High (>1000)
0.6 image + 0.4 fusion
✅ Reduces SMAPE by 5–10 points.


Phase 6: Validation & Error Analysis (Days 21–22)
🔍 6.1 SMAPE Analysis
Fold-wise SMAPE:
 
smape = 200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))

Visualize by price bin, category, brand
⚠️ 6.2 Weak Segment Identification
Poor text → rely on image
Missing image → rely on text/brand cues
Outliers → clip to [P5, P95]
✅ Improves consistency & reduces SMAPE variance.

Phase 7: Final Submission (Day 23)
🏁 7.1 Test Predictions
Generate ensemble predictions
Inverse log:
 
preds = np.expm1(final_preds)

Clip to train price bounds
📦 7.2 Deliverables
submission.csv
requirements.txt
model_card.md
Pipeline: train.py, predict.py

🧱 Recommended Tech Stack
Type
Libraries
Core ML
PyTorch, LightGBM, XGBoost, CatBoost
Text
NLTK, spaCy, SentenceTransformers
Image
timm, OpenCV, torchvision
Optimization
Optuna
EDA
pandas, matplotlib, seaborn

🚀 Target Score Progression (Optimized)
Stage
Expected SMAPE
Key Actions
Baseline LightGBM
58.6
Structured features + log-transform
+ Text & Semantic Features
50–53
TF-IDF, embeddings, keyword/sentiment flags
+ Image & Basic Fusion
46–49
EfficientNet/CLIP embeddings, PCA + concatenation
+ Dual-Tower Alignment
43–45
Contrastive learning / text-image cross-attention
+ Cohort-Aware Tuning
40–42
GroupKFold, custom loss, hyperparameter optimization
+ Two-Stage Ensemble & Post-Process
36–38 ✅
Stacking meta-learner, dynamic blending, bias calibration
