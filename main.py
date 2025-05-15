# Iimporting essential libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import joblib


# Loading the dataset
df = pd.read_csv('Dataset.csv')
print(df.head(10))
# Returns number of rows and columns of the dataset
print(df.shape)

# Data Preprocessing
# Data cleaning
null_values = df.isnull().sum()
print('Data Cleaned Result:')
print(null_values)

# Data Visulaization

# Set plot style
sns.set_style("whitegrid")

# 1. Histogram - Soil Moisture Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Soil Moisture (%)'], bins=30, kde=True, color='blue')
plt.title('Distribution of Soil Moisture Levels')
plt.xlabel('Soil Moisture (%)')
plt.ylabel('Frequency')

# 2. Box Plot - Vibration Intensity
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Landslide'], y=df['Vibration Intensity'], palette='coolwarm')
plt.title('Vibration Intensity Distribution')
plt.xlabel('Landslide')
plt.ylabel('Vibration Intensity')
plt.xticks([0, 1], ['Stable', 'Landslide'])


# 3. Pie Chart - Landslide vs. Normal Cases
plt.figure(figsize=(5, 5))
labels = ['Stable', 'Landslide']
sizes = df['Landslide'].value_counts()
colors = ['lightblue', 'red']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, shadow=True)
plt.title('Landslide Occurrence Distribution')

# Data Scaling using RobustScaler Transform (Data Normalization)

X = df.drop('Landslide', axis=1)
y = df['Landslide']

sc = RobustScaler()
X = sc.fit_transform(X)
print('Data Normalized Result:')
X = pd.DataFrame(X)
print(X.head())

# Save the trained scaler using joblib
joblib.dump(sc, 'scaler.pkl')

# Dataset splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print('training set size: {}, testing set size: {}'.format(X_train.shape, X_test.shape))

# Ensemble ML Model Build

# Define models
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
rf = RandomForestClassifier()
gbm = GradientBoostingClassifier()

# Define hyperparameter grids for each model
xgb_params = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
}

rf_params = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}

gbm_params = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Run RandomizedSearchCV for each model
xgb_random = RandomizedSearchCV(xgb, xgb_params, n_iter=10, scoring='accuracy', n_jobs=-1, cv=5, random_state=42, verbose=3)
rf_random = RandomizedSearchCV(rf, rf_params, n_iter=10, scoring='accuracy', n_jobs=-1, cv=5, random_state=42, verbose=3)
gbm_random = RandomizedSearchCV(gbm, gbm_params, n_iter=10, scoring='accuracy', n_jobs=-1, cv=5, random_state=42, verbose=3)

# Fit models on training data
xgb_random.fit(X_train, y_train)
rf_random.fit(X_train, y_train)
gbm_random.fit(X_train, y_train)

# Get the best estimators
best_xgb = xgb_random.best_estimator_
best_rf = rf_random.best_estimator_
best_gbm = gbm_random.best_estimator_

# Print best hyperparameters
print("\nBest Hyperparameters for XGBoost:\n", best_xgb)
print("\nBest Hyperparameters for Random Forest:\n", best_rf)
print("\nBest Hyperparameters for Gradient Boosting:\n", best_gbm)

# Visualization of Hyperparameter Tuning Results
def plot_heatmap(results_df, title, index_col, column_col, value_col):
    pivot = results_df.pivot_table(index=index_col, columns=column_col, values=value_col)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".3f", cbar=True)
    plt.title(title)
    plt.xlabel(column_col)
    plt.ylabel(index_col)

# Extract hyperparameter results
xgb_results_df = pd.DataFrame(xgb_random.cv_results_)
rf_results_df = pd.DataFrame(rf_random.cv_results_)
gbm_results_df = pd.DataFrame(gbm_random.cv_results_)

# Plot heatmaps
plot_heatmap(xgb_results_df, 'XGBoost Mean Test Score Heatmap', 'param_n_estimators', 'param_max_depth', 'mean_test_score')
plot_heatmap(rf_results_df, 'Random Forest Mean Test Score Heatmap', 'param_n_estimators', 'param_max_depth', 'mean_test_score')
plot_heatmap(gbm_results_df, 'Gradient Boosting Mean Test Score Heatmap', 'param_n_estimators', 'param_max_depth', 'mean_test_score')

# Define the Ensemble Model using VotingClassifier
eml_model = VotingClassifier(
    estimators=[('xgb', best_xgb), ('rf', best_rf), ('gbm', best_gbm)],
    voting='soft'  # Soft voting for probabilistic averaging
)

print("\nSuccessfully Built EML Model with RF, XGBoost, and Gradient Boosting!")

# Train the EML model

eml_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(eml_model, "eml_model.pkl")

# Testing

y_pred = eml_model.predict(X_test)
X_test = pd.DataFrame(X_test)
y_pred = np.round(y_pred)

df = pd.DataFrame(y_test)
df['Predicted Result']=y_pred
df = df.rename(columns={'Landslide': 'Actual Result'})
df

# Creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)
# Plotting the confusion matrix
plt.figure(figsize=(7,5))
lang=['Stable','Landslide']
cm = pd.DataFrame(cm,columns=lang,index=lang)
p = sns.heatmap(cm, annot=True, fmt='g')
plt.title('Confusion matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Evaluation

# Accuracy
score = round(accuracy_score(y_test, y_pred),4)*100
print("Accuracy on test set: {}%".format(score))

# Precision, Recall and F1-Score

print(classification_report(y_test, y_pred, digits=4))

# ROC Curve

test_preds_prob = eml_model.predict_proba(X_test)[:, 1]

test_trues = np.array(y_test)

# Compute ROC curve
fpr, tpr, thresholds = metrics.roc_curve(test_trues, test_preds_prob)
roc_auc = metrics.auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'EML Model (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc='lower right')
plt.grid(False)
plt.show()
