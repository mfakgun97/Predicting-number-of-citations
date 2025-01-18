## Paper Citation Prediction with Ridge, ElasticNet, XGBoost, and Stacked Models

TiU Data Science MAster Machine Learning Individual Assignment Project.
This work graded based on the Codalab competition created by the teachers.

# Overview

This project predicts the number of citations (n_citation) for academic papers based on metadata such as the paper's year, authors, title, venue, and abstract. The final model achieves a Mean Absolute Error (MAE) of 30.56, indicating good predictive performance on the validation dataset.



This repository contains a Python script that trains multiple regression models, fine-tunes hyperparameters, and generates predictions for a target variable (n_citation). The script utilizes machine learning techniques to process data and create robust predictions.

# Workflow

1. Data Loading
Data is loaded from train.json and test.json, which are structured as JSON files.
The train dataset is split into training and validation sets in a 2:1 ratio for model evaluation.
PS: Original datasets contain -> test.json : 396404 and train.json : 1189209 items

2. Feature Engineering
A ColumnTransformer is used to preprocess the features:

Year: Passed directly without transformation.
Textual Features: (authors, title, venue, abstract)
Transformed using TfidfVectorizer:
n-grams: Bi-grams (n=1-2) to capture word sequences.
Max Features: Limits vocabulary size to focus on the most important terms.
Min DF: Ignores terms appearing in fewer than 3 documents.
This feature engineering pipeline ensures numerical and textual data are combined effectively for modeling.

3. Model Selection
Three base models and a stacking ensemble are implemented:

Ridge Regression:
Ridge applies L2 regularization to prevent overfitting.
The best regularization parameter alpha is selected through random sampling from a log-uniform distribution.
Validation MAE for each alpha is logged, and the best value is chosen.
ElasticNet:
Combines L1 (lasso) and L2 (ridge) regularization.
Uses the best alpha from Ridge, with l1_ratio=0.5 for balanced regularization.
XGBoost:
A powerful gradient boosting algorithm using decision trees.
Configured with:
100 estimators.
Max tree depth of 5.
Learning rate of 0.1.
Stacked Regressor:
Combines the predictions from Ridge, ElasticNet, and XGBoost using another Ridge model as the meta-learner.
The stacking approach leverages the strengths of all models for improved predictions.

4. Training and Evaluation
Each model is trained on the training dataset and evaluated on the validation dataset.
Metrics:
Train MAE: Measures model performance on the training data.
Validation MAE: Measures generalization performance.
Results are logged for comparison.

5. Predictions
The stacked model is retrained on the entire training dataset.
Predictions on the test dataset are generated and saved to predicted.json.

6. Submission File
The predictions are zipped into submission.zip for easy sharing.

## Key Highlights

# Feature Engineering:
Captures both numerical and textual data effectively using TF-IDF.
# Hyperparameter Tuning:
Uses random sampling for Ridge regularization (alpha) to optimize performance.
# Ensemble Modeling:
Combines multiple models with diverse strengths to achieve robust predictions.
# MAE:
Achieves 30.56 MAE, which reflects the model's ability to predict citations accurately.

# Results

The stacked ensemble model outperforms individual models, demonstrating the effectiveness of combining different algorithms.

