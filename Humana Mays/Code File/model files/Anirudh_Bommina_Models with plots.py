#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load datasets
target_train = pd.read_csv('D:\\HUMANA CASE COMPI\\target_train.csv')
medclms_train = pd.read_csv('D:\\HUMANA CASE COMPI\\medclms_train.csv')
rxclms_train = pd.read_csv('D:\\HUMANA CASE COMPI\\rxclms_train.csv')

# Preprocessing for target_train dataset
target_train['race_cd'].fillna(target_train['race_cd'].mode()[0], inplace=True)
target_train['est_age'].fillna(target_train['est_age'].median(), inplace=True)
target_train['sex_cd'].fillna(target_train['sex_cd'].mode()[0], inplace=True)
target_train['cms_disabled_ind'].fillna(target_train['cms_disabled_ind'].mode()[0], inplace=True)
target_train['cms_low_income_ind'].fillna(target_train['cms_low_income_ind'].mode()[0], inplace=True)

# Make sure therapy_start_date and therapy_end_date are timezone-aware
target_train['therapy_start_date'] = pd.to_datetime(target_train['therapy_start_date']).dt.tz_localize(None)
target_train['therapy_end_date'] = pd.to_datetime(target_train['therapy_end_date']).dt.tz_localize(None)
target_train['therapy_duration'] = (target_train['therapy_end_date'] - target_train['therapy_start_date']).dt.days
target_train = pd.get_dummies(target_train, columns=['sex_cd'], drop_first=True)

# Preprocessing for medclms_train dataset
cols_to_fill_unknown_med = ['diag_cd2', 'diag_cd3', 'diag_cd4', 'diag_cd5', 'diag_cd6', 
                            'diag_cd7', 'diag_cd8', 'diag_cd9', 'reversal_ind', 'util_cat']
for col in cols_to_fill_unknown_med:
    medclms_train[col].fillna("Unknown", inplace=True)
medclms_train['visit_date'] = pd.to_datetime(medclms_train['visit_date'])
medclms_train['process_date'] = pd.to_datetime(medclms_train['process_date'])

# Preprocessing for rxclms_train dataset
cols_to_fill_unknown_rx = ['gpi_drug_group_desc', 'gpi_drug_class_desc', 'hum_drug_class_desc', 'strength_meas']
for col in cols_to_fill_unknown_rx:
    rxclms_train[col].fillna("Unknown", inplace=True)
rxclms_train['metric_strength'].fillna(rxclms_train['metric_strength'].median(), inplace=True)
rxclms_train['service_date'] = pd.to_datetime(rxclms_train['service_date'])
rxclms_train['process_date'] = pd.to_datetime(rxclms_train['process_date'])



# Aggregated features for medclms_train
num_unique_diagnoses = medclms_train.groupby('therapy_id')['primary_diag_cd'].nunique().reset_index()
num_unique_diagnoses.rename(columns={'primary_diag_cd': 'num_unique_diagnoses'}, inplace=True)
total_ade_diagnoses = medclms_train.groupby('therapy_id')['ade_diagnosis'].sum().reset_index()
total_ade_diagnoses.rename(columns={'ade_diagnosis': 'total_ade_diagnoses'}, inplace=True)

# Aggregated features for rxclms_train
total_rx_cost = rxclms_train.groupby('therapy_id')['rx_cost'].sum().reset_index()
total_rx_cost.rename(columns={'rx_cost': 'total_rx_cost'}, inplace=True)
total_ddi_indications = rxclms_train.groupby('therapy_id')['ddi_ind'].sum().reset_index()
total_ddi_indications.rename(columns={'ddi_ind': 'total_ddi_indications'}, inplace=True)

# Merge aggregated features with target_train
target_train = pd.merge(target_train, num_unique_diagnoses, on='therapy_id', how='left')
target_train = pd.merge(target_train, total_ade_diagnoses, on='therapy_id', how='left')
target_train = pd.merge(target_train, total_rx_cost, on='therapy_id', how='left')
target_train = pd.merge(target_train, total_ddi_indications, on='therapy_id', how='left')
target_train.fillna(0, inplace=True)

# Split the data into features and target
X = target_train.drop(columns=['id', 'therapy_id', 'therapy_start_date', 'therapy_end_date', 'tgt_ade_dc_ind'])
y = target_train['tgt_ade_dc_ind']

# Initialize models
log_reg = LogisticRegression(max_iter=10000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gboost = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42)

models = [log_reg, rf, gboost, xgb]

# Initialize lists to store ROC AUC scores
roc_auc_scores = []
# Initialize lists to store confusion matrices
confusion_matrices = []

# Perform cross-validation and compute confusion matrices
for model in models:
    y_pred_cv = cross_val_predict(model, X, y, cv=5)
    cm = confusion_matrix(y, y_pred_cv)
    confusion_matrices.append(cm)

# Perform cross-validation
for model in models:
    y_prob_cv = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
    roc_auc = roc_auc_score(y, y_prob_cv)
    roc_auc_scores.append(roc_auc)
    

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob_cv)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model.__class__.__name__}')
    plt.legend(loc='lower right')
    plt.show()

    # Print ROC AUC score
    print(f'{model.__class__.__name__}: ROC AUC Score: {roc_auc:.4f}')

# Print evaluation results
for model, roc_auc in zip(models, roc_auc_scores):
    print(f'{model.__class__.__name__}: ROC AUC Score: {roc_auc:.4f}')

# %%
# Print confusion matrices
for model, cm in zip(models, confusion_matrices):
    print(f'{model.__class__.__name__} Confusion Matrix:\n{cm}')

# %%
# Initialize LogisticRegression
logistic_model = LogisticRegression(max_iter=10000, random_state=42)

# Train the logistic regression model
logistic_model.fit(X, y)
# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load holdout datasets
target_holdout = pd.read_csv('D:\\HUMANA CASE COMPI\\target_holdout.csv')
medclms_holdout = pd.read_csv('D:\\HUMANA CASE COMPI\\medclms_holdout.csv')
rxclms_holdout = pd.read_csv('D:\\HUMANA CASE COMPI\\rxclms_holdout.csv')

# Preprocess target_holdout.csv
target_holdout = pd.get_dummies(target_holdout, columns=['sex_cd'], drop_first=True)

# Preprocess medclms_holdout.csv
medclms_holdout.drop(columns=['reversal_ind', 'clm_type'], inplace=True)
medclms_holdout = pd.get_dummies(medclms_holdout, columns=['pot', 'util_cat', 'hedis_pot'], drop_first=True)

# Convert 'visit_date' to datetime format
medclms_holdout['visit_date'] = pd.to_datetime(medclms_holdout['visit_date'])

# Now you can use .dt accessor
medclms_holdout['visit_day_of_week'] = medclms_holdout['visit_date'].dt.dayofweek
medclms_holdout['visit_month'] = medclms_holdout['visit_date'].dt.month

# Preprocess rxclms_holdout.csv
rxclms_holdout.drop(columns=['reversal_ind', 'clm_type', 'document_key'], inplace=True)
rxclms_holdout = pd.get_dummies(rxclms_holdout, columns=['gpi_drug_group_desc', 'gpi_drug_class_desc', 'hum_drug_class_desc'], drop_first=True)
rxclms_holdout['service_date'] = pd.to_datetime(rxclms_holdout['service_date'])  # Convert 'service_date' to datetime format
rxclms_holdout['service_day_of_week'] = rxclms_holdout['service_date'].dt.dayofweek
rxclms_holdout['service_month'] = rxclms_holdout['service_date'].dt.month


# Aggregated features for medclms_holdout.csv
medclms_count_holdout = medclms_holdout.groupby('therapy_id').size().reset_index(name='medclms_count')
medclms_diagnosis_sum_holdout = medclms_holdout.groupby('therapy_id')[['ade_diagnosis', 'seizure_diagnosis']].sum().reset_index()
medclms_aggregated_holdout = pd.merge(medclms_count_holdout, medclms_diagnosis_sum_holdout, on='therapy_id')

# Aggregated features for rxclms_holdout.csv
rxclms_count_holdout = rxclms_holdout.groupby('therapy_id').size().reset_index(name='rxclms_count')
rxclms_avg_cost_holdout = rxclms_holdout.groupby('therapy_id')['rx_cost'].mean().reset_index(name='avg_rx_cost')
rxclms_treatment_sum_holdout = rxclms_holdout.groupby('therapy_id')[['ddi_ind', 'anticoag_ind', 'diarrhea_treat_ind']].sum().reset_index()
rxclms_aggregated_holdout = pd.merge(rxclms_count_holdout, rxclms_avg_cost_holdout, on='therapy_id')
rxclms_aggregated_holdout = pd.merge(rxclms_aggregated_holdout, rxclms_treatment_sum_holdout, on='therapy_id')

# Merge the datasets
consolidated_data_holdout = pd.merge(target_holdout, medclms_aggregated_holdout, on='therapy_id', how='left')
consolidated_data_holdout = pd.merge(consolidated_data_holdout, rxclms_aggregated_holdout, on='therapy_id', how='left')
consolidated_data_holdout.fillna(0, inplace=True)

# Align columns of holdout data with training data
X_holdout = consolidated_data_holdout.drop(columns=['id', 'therapy_id', 'therapy_start_date'])
X_holdout = X_holdout.reindex(columns=X.columns, fill_value=0) 

# Predict probabilities for holdout data using the logistic regression model
proba_scores = logistic_model.predict_proba(X_holdout)[:, 1]

# Create DataFrame with ID and predicted probabilities
output_df = pd.DataFrame({
    'ID': consolidated_data_holdout['id'],
    'score': proba_scores
})
output_df['rank'] = output_df['score'].rank(ascending=False).astype(int)
#%%
output_df.to_csv('output.csv', index=False)


# %%
