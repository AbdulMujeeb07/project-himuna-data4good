#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load your dataset
data = pd.read_csv('D:\\HUMANA CASE COMPI\\merged_data.csv')

# Remove non-numeric columns
non_numeric_cols = ['therapy_id', 'therapy_start_date', 'therapy_end_date']
data = data.drop(non_numeric_cols, axis=1)

# Encode 
label_encoder = LabelEncoder()
data['race_cd'] = label_encoder.fit_transform(data['race_cd'])
data['sex_cd_M'] = label_encoder.fit_transform(data['sex_cd_M'])

# training and testing sets
X = data.drop('tgt_ade_dc_ind', axis=1)
y = data['tgt_ade_dc_ind']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  parameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
}

# Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=42)

# grid search for hyperparameter tuning
grid_search = GridSearchCV(gb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_gb_classifier = grid_search.best_estimator_

# Model 
y_pred = best_gb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print('Gradiant Boost Model Score:')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Confusion Matrix:\n{conf_matrix}')


