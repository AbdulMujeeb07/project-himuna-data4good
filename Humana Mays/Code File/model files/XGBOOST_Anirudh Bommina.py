
#%%
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

#dataset
data = pd.read_csv('D:\\HUMANA CASE COMPI\\merged_data.csv')

# Remove non-numeric columns
non_numeric_cols = ['therapy_id', 'therapy_start_date', 'therapy_end_date']
data = data.drop(non_numeric_cols, axis=1)

# Encode categorical variables 
label_encoder = LabelEncoder()
data['race_cd'] = label_encoder.fit_transform(data['race_cd'])
data['sex_cd_M'] = label_encoder.fit_transform(data['sex_cd_M'])

#  dataset into training and testing sets
X = data[['race_cd', 'est_age', 'cms_disabled_ind', 'cms_low_income_ind', 'therapy_duration', 'sex_cd_M', 'num_unique_diagnoses', 'total_ade_diagnoses', 'total_rx_cost', 'total_ddi_indications']]
y = data['tgt_ade_dc_ind']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)

# extended parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
}

#  hyperparameter tuning
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# XGBoost classifier 
best_xgb_classifier = grid_search.best_estimator_

# Predict probabilities on the test set
y_probs = best_xgb_classifier.predict_proba(X_test)[:, 1]

# Define thresholds for evaluation
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

#  best threshold and ROC AUC score
best_threshold = None
best_roc_auc = 0.0

# Iterate through thresholds and evaluate the model
for threshold in thresholds:
    y_pred = (y_probs >= threshold).astype(int)
    roc_auc = roc_auc_score(y_test, y_pred)
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_threshold = threshold

#  best threshold
y_pred = (y_probs >= best_threshold).astype(int)

#  evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# valuation metrics
print('XGBoost Model Score:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {best_roc_auc:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print(f'Best Threshold for ROC AUC: {best_threshold:.2f}')


# %%
