# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:41:41 2024

@author: smid
"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer

# Step 1: Load the Data
train_df = pd.read_csv('train.csv')

# Step 2: Handle Missing Data
# Create flags for missing data
for column in train_df.columns:
    if train_df[column].isnull().sum() > 0:
        train_df[f'{column}_missing'] = train_df[column].isnull().astype(int)

# Impute missing values
# Numeric features: Impute with median
numerical_features = ['cap-diameter', 'stem-height', 'stem-width']
for column in numerical_features:
    if train_df[column].isnull().sum() > 0:
        train_df[column].fillna(train_df[column].median(), inplace=True)

# Categorical features: Impute with mode
categorical_features = [col for col in train_df.columns if col not in numerical_features + ['id', 'class'] and '_missing' not in col]
for column in categorical_features:
    if train_df[column].isnull().sum() > 0:
        train_df[column].fillna(train_df[column].mode()[0], inplace=True)

# Step 3: Encode Target Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df['class'])  # Encode 'e' as 0 and 'p' as 1

# Step 4: Preprocess the Data
# Define a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Drop 'class' and 'id' columns, retain others including the _missing flags
X = train_df.drop(columns=['class', 'id'])

# Transform the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Step 5: Hyperparameter Tuning with XGBoost and Early Stopping
# Split training data further into a training set and an early stopping validation set
X_train_sub, X_stop, y_train_sub, y_stop = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define the XGBoost model
xgb_model = XGBClassifier(random_state=42, n_estimators=1000)  # Set high n_estimators to allow for early stopping

# Define the parameter grid for Grid Search
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Set up cross-validation configuration
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use GridSearchCV to tune hyperparameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring=make_scorer(matthews_corrcoef), 
                           cv=cv, verbose=2, n_jobs=-1)

# Fit the model with Grid Search and Early Stopping
grid_search.fit(X_train_sub, y_train_sub, 
                eval_set=[(X_stop, y_stop)],
                eval_metric="logloss",  # Use a valid metric for classification
                early_stopping_rounds=10,  # Stops if performance does not improve for 10 rounds
                verbose=True)

# Get the best parameters and evaluate on the validation set
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Predict on the validation set
y_pred = best_model.predict(X_val)
mcc = matthews_corrcoef(y_val, y_pred)
print(f"Validation MCC (XGBoost with Early Stopping): {mcc}")

# Step 6: Cross-Validation (using best model)
# Evaluate using cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=make_scorer(matthews_corrcoef))
print(f"Cross-Validation MCC Scores: {cv_scores}")
print(f"Average MCC: {cv_scores.mean()}")

# Step 7: Stacking Ensemble
# Define base models for stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', best_model)
]

# Define the stacking ensemble with Logistic Regression as the final estimator
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)

# Fit the stacking model
stacking_model.fit(X_train, y_train)

# Evaluate the stacked model on the validation set
y_pred_stack = stacking_model.predict(X_val)
mcc_stack = matthews_corrcoef(y_val, y_pred_stack)
print(f"Validation MCC (Stacking): {mcc_stack}")

# Step 8: Final Model Evaluation and Prediction on Test Set
# Use the best-performing model to generate predictions for the test set
# Load test data
test_df = pd.read_csv('test.csv')

# Apply the same preprocessing
X_test = preprocessor.transform(test_df.drop(columns=['id']))

# Predict using the stacking model (or best model)
test_predictions = stacking_model.predict(X_test)

# Encode predictions back to original labels
test_predictions = label_encoder.inverse_transform(test_predictions)

# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'class': test_predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)
