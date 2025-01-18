


pip install xgboost





import pandas as pd
import numpy as np
import logging
import json
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor

def main():
    
    train = pd.DataFrame.from_records(json.load(open('train.json')))
    test = pd.DataFrame.from_records(json.load(open('test.json')))

    
    train, validation = train_test_split(train, test_size=1/3, random_state=123)

    
    featurizer = ColumnTransformer(
        transformers=[
            ("year", "passthrough", ["year"]),
            ("authors", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3), "authors"),
            ("title", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3), "title"),
            ("venue", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3), "venue"),
            ("abstract", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=3), "abstract"),
        ],
        remainder="drop"
    )

    label = 'n_citation'

    
    random.seed(42)
    n_iter = 20
    alpha_candidates = [10 ** random.uniform(-3, 3) for _ in range(n_iter)]
    best_alpha = None
    best_mae = float('inf')

    for alpha in alpha_candidates:
        ridge = make_pipeline(featurizer, Ridge(alpha=alpha))
        ridge.fit(train.drop([label], axis=1), np.log1p(train[label].values))
        pred = np.expm1(ridge.predict(validation.drop([label], axis=1)))
        mae = mean_absolute_error(validation[label], pred)

        logging.info(f"Ridge Alpha: {alpha:.5f}, Validation MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    logging.info(f"Best Ridge Alpha: {best_alpha:.5f}, Best Validation MAE: {best_mae:.2f}")

    
    ridge_model = make_pipeline(featurizer, Ridge(alpha=best_alpha))

    
    elasticnet_model = make_pipeline(featurizer, ElasticNet(alpha=best_alpha, l1_ratio=0.5))

    
    xgboost_model = make_pipeline(featurizer, XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1))

    
    stacked_model = StackingRegressor(
        estimators=[
            ("ridge", ridge_model),
            ("elasticnet", elasticnet_model),
            ("xgboost", xgboost_model),
        ],
        final_estimator=Ridge(alpha=1)
    )

    
    for model_name, model in [("Ridge", ridge_model), ("ElasticNet", elasticnet_model), ("XGBoost", xgboost_model), ("Stacked", stacked_model)]:
        logging.info(f"Fitting model {model_name}")
        model.fit(train.drop([label], axis=1), np.log1p(train[label].values))

    
        
        train_pred = np.expm1(model.predict(train.drop([label], axis=1)))
        mae_train = mean_absolute_error(train[label], train_pred)
        logging.info(f"{model_name} Train MAE: {mae_train:.2f}")
    
        
        validation_pred = np.expm1(model.predict(validation.drop([label], axis=1)))
        mae_validation = mean_absolute_error(validation[label], validation_pred)
        logging.info(f"{model_name} Validation MAE: {mae_validation:.2f}")

        

    
    stacked_model.fit(train.drop([label], axis=1), np.log1p(train[label].values))
    test_predictions = np.expm1(stacked_model.predict(test))

    
    test['n_citation'] = test_predictions
    predictions = test[['n_citation']].to_dict(orient='records')

    with open('predicted.json', 'w') as f:
        json.dump(predictions, f, indent=2)

    
    import zipfile
    with zipfile.ZipFile('submission.zip', 'w') as zipf:
        zipf.write('predicted.json')

    logging.info("Submission file created: submission.zip")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()







