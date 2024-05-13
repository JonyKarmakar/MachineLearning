# -*- coding: utf-8 -*-
"""CA4_final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tTwxPI5g3v-L217uRA8rDMhnQS0Ta-zH

# CA4 - Liver Disease Prediction
### Jony Karmakar

### Imports
"""

# Importing Libaries
# ======================
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

"""### Reading data"""

# Loading the dataset
# ======================
df_train = pd.read_csv('train.csv', index_col=0)
df_test = pd.read_csv('test.csv', index_col=0)

"""### Data exploration and visualisation"""

#Cheking for missing data
# =======================
null_counts = df_train.isnull().sum().sum()
print(f'Total Missing values: {null_counts}' )

# Checking the info of the dataset
# ================================
df_train.info()

# Printing the first 10 rows of the dataframe with head(10)
# =========================================================
df_train.head(10)

# Dropping the first column
# ==========================
df_train = df_train.drop('index', axis=1)

# Shwowing descriptive statistics of the dataset
# ==============================================
df_train.describe()

"""### Histogram"""

# Plotting histograms for all columns in the dataset
# ==================================================
df_train.hist(bins=10, figsize=(26, 15))
plt.show()

# Removing the target variable
# =============================
features = df_train.columns[:-1]

plt.figure(figsize=(20, 60))

for i, column in enumerate(features):
    plt.subplot(len(features), 4, i + 1)
    sns.boxplot(y=df_train[column])
    plt.title(column)
    plt.tight_layout()

plt.show()

"""#### Violin Plots"""

# Removing the target variable
# =============================
features = df_train.columns[:-1]

plt.figure(figsize=(20, 60))

for i, column in enumerate(features):
    plt.subplot(len(features), 4, i + 1)
    sns.violinplot(y=df_train[column])
    plt.title(column)

plt.tight_layout()
plt.show()

"""### Data cleaning"""

# Coping the dataset
# ==================
df_train_copy = df_train.copy()

# printing the columns of the dataset
# ====================================
df_train_copy.info()

"""#### Handling Catagorical Features"""

# Storing item of Diagnosis columns
# =================================
diagonosis = df_train_copy['Diagnosis'].unique()

# Creating dictionary from array for diagnosis
# ============================================
diagonosis_dict = dict(zip(diagonosis, range(len(diagonosis))))

# Encoding the class label(Diagnosis) to integers
# ===============================================
df_train_copy['Diagnosis'] = df_train_copy['Diagnosis'].map(diagonosis_dict)

# Converting the catagorical data to numerical data
# =================================================
df_dummies = pd.get_dummies(df_train_copy, drop_first=True)

# Checking the info of the dataset
# ================================
df_dummies.info()

# Moving the target variable to the last column
# =============================================
df_dummies = df_dummies[[col for col in df_dummies.columns if col != 'Diagnosis'] + ['Diagnosis']]
df_dummies.head()

"""#### Removing Outliers"""

# Removing the outliers from the dataset using IQR method only for numerical data
# ================================================================================
# Taking only the numerical columns
df_numerical = df_dummies.select_dtypes(include=[np.number])
# Excluding the target variable
df_numerical = df_numerical.drop('Diagnosis', axis=1)
# Calculating the IQR
Q1 = df_numerical.quantile(0.25)
Q3 = df_numerical.quantile(0.75)
IQR = Q3 - Q1
# Removing the outliers
df_numerical_2 = df_numerical[~((df_numerical < (Q1 - 1.5 * IQR)) | (df_numerical > (Q3 + 1.5 * IQR))).any(axis=1)]
df_numerical_2.info()

"""##### Using IQR method almost half of the data is getting eliminated so this method doesn't seems good one.
##### In the below I tried the z-score method to clean the data
"""

# Detecting the outliers using the z-score (With the help of lecture note)
# =========================================================================
for column in df_numerical.columns:
    # Calculating the Z-scores for each column
    z_scores = (df_numerical[column] - df_numerical[column].mean()) / df_numerical[column].std()

    # Detecting outliers using the absolute value of the Z-scores (threshold of 2)
    outliers = (np.abs(z_scores) > 2)

    print(f"Number of outliers in {column}: {outliers.sum()}")

# Printing information of the dataset with numerical columns
# ==========================================================
df_numerical.info()

# Removing the outliers from the dataset using Z-score method only for numerical data (With the help of lecture note)
# ===================================================================================================================
for column in df_numerical.columns:
    # Calculate the z-scores for each column
    z_scores = (df_numerical[column] - df_numerical[column].mean()) / df_numerical[column].std()

    # Only keep rows in dataframe where the z-score is less than 2 standard deviations
    df_numerical_3 = df_numerical[np.abs(z_scores) < 2]

# Checking the information of the dataset
df_numerical_3.info()

"""##### This seems little bit more promising than the IQR. In the next steps both datasets with outliers and without outliers will be used."""

# Merging the numerical columns with the catagorical columns based on the index column
# ====================================================================================
rest_columns = ['Alcohol_Use (yes/no)_yes', 'Diabetes (yes/no)_yes',
                    'Gender_MALE', 'Obesity (yes/no)_yes', 'Diagnosis']
df_cleaned = pd.merge(df_numerical_3, df_dummies[rest_columns], left_index=True, right_index=True)
df_cleaned.info()

"""#### Visualizing after cleaning"""

# Removing the target variable
# =============================
features = df_cleaned.columns[:-1]

plt.figure(figsize=(20, 60))

for i, column in enumerate(features):
    plt.subplot(len(features), 4, i + 1)
    sns.boxplot(y=df_cleaned[column])
    plt.title(column)
    plt.tight_layout()

plt.show()

"""##### Still shows a lot of Outliers.

### Data preprocessing and visualisation

#### Splitting the Dataset
"""

#-------------------------------------------------
# Spliting the cleaned dataset into test and train
#-------------------------------------------------
X_cleaned = df_cleaned.drop('Diagnosis', axis=1)
y_cleaned = df_cleaned['Diagnosis']

X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

#------------------------------------------------------
# Spliting the dataset with outlier into test and train
#------------------------------------------------------
X = df_dummies.drop('Diagnosis', axis=1)
y = df_dummies['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#### Exploring training data after cleaning and splitting"""

X_cleaned.info()

# Plotting histograms for all columns in the dataset
# ==================================================
X_cleaned.hist(bins=10, figsize=(26, 15))
plt.show()

# Plotting the Boxplot for the cleaned dataset i.e X_train_cleaned_sc_df
# ======================================================================
features = X_cleaned.columns[:-1]

plt.figure(figsize=(20, 60))

for i, column in enumerate(features):
    plt.subplot(len(features), 4, i + 1)
    sns.boxplot(y=X_cleaned[column])
    plt.title(column)
    plt.tight_layout()

plt.show()

# Plotting the Violin plot for the cleaned dataset i.e X_train_cleaned_sc_df
# ==========================================================================
features = X_cleaned.columns[:-1]

plt.figure(figsize=(20, 60))

for i, column in enumerate(features):
    plt.subplot(len(features), 4, i + 1)
    sns.violinplot(y=X_cleaned[column])
    plt.title(column)

plt.tight_layout()
plt.show()

"""### Modelling

#### Pipelining with Kernel
"""

# Defining the pipeline with kernel(SVC) and pca
# ==============================================
pipe_svc_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', SVC())
])

# Defining the parameters for the grid search
# ===========================================
param_grid_svc_pca = {
    'pca__n_components': [0.9, 0.95, 0.99],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Defining the grid search
# ========================
grid_search_svc_pca = GridSearchCV(estimator=pipe_svc_pca, param_grid=param_grid_svc_pca, scoring='f1_macro', cv=5)

# Fitting the grid search
# =======================
grid_search_svc_pca.fit(X_train_cleaned, y_train_cleaned)

# Printing the best parameters
# ============================
print(f"Best parameters: {grid_search_svc_pca.best_params_}")

# Printing the best score
# ========================
print(f"Best score: {grid_search_svc_pca.best_score_}")

# Defining the pipeline with kernel(SVC) and lda
# ==============================================
pipe_svc_lda = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA()),
    ('classifier', SVC())
])

# Defining the parameters for the grid search
# ===========================================
param_grid_svc_lda= {
    'lda__n_components': [1, 2],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Defining the grid search
# ========================
grid_search_svc_lda = GridSearchCV(estimator=pipe_svc_lda, param_grid=param_grid_svc_lda, scoring='f1_macro', cv=5)

# Fitting the grid search
# =======================
grid_search_svc_lda.fit(X_train_cleaned, y_train_cleaned)

# Printing the best parameters
# ============================
print(f"Best parameters: {grid_search_svc_lda.best_params_}")

# Printing the best score
# ========================
print(f"Best score: {grid_search_svc_lda.best_score_}")

"""##### Pipelining with Regularization"""

# Defining the pipeline with Regulization(LogisticRegression) and pca
# ===================================================================
pipe_lr_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', LogisticRegression())
])

# Defining the parameters for the grid search
# ===========================================
param_grid_lr_pca = {
    'pca__n_components': [0.9, 0.95, 0.99],
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2']
}

# Defining the grid search
# ========================
grid_search_lr_pca = GridSearchCV(estimator=pipe_lr_pca, param_grid=param_grid_lr_pca, scoring='f1_macro', cv=5)

# Fitting the grid search
# =======================
grid_search_lr_pca.fit(X_train_cleaned, y_train_cleaned)

# Printing the best parameters
# ============================
print(f"Best parameters: {grid_search_lr_pca.best_params_}")

# Printing the best score
# ========================
print(f"Best score: {grid_search_lr_pca.best_score_}")

# Defining the pipeline with Regulization(LogisticRegression) and lda
# ===================================================================
pipe_lr_lda = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA()),
    ('classifier', LogisticRegression())
])

# Defining the parameters for the grid search
# ===========================================
param_grid_lr_lda = {
    'lda__n_components': [1, 2],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

# Defining the grid search
# ========================
grid_search_lr_lda = GridSearchCV(estimator=pipe_lr_lda, param_grid=param_grid_lr_lda, scoring='f1_macro', cv=5)

# Fitting the grid search
# =======================
grid_search_lr_lda.fit(X_train_cleaned, y_train_cleaned)

# Printing the best parameters
# ============================
print(f"Best parameters: {grid_search_lr_lda.best_params_}")

# Printing the best score
# ========================
print(f"Best score: {grid_search_lr_lda.best_score_}")

# Defining the pipeline with RandomForestClassifier and pca
# =========================================================
pipe_rf_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier())
])

# Defining the parameters for the grid search with f1 macro average as scoring
# ============================================================================
param_grid_rf_pca = {
    'pca__n_components': [0.9, 0.95, 0.99],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, 20],
}

# Defining the grid search
# ========================
grid_search_rf_pca = GridSearchCV(estimator=pipe_rf_pca, param_grid=param_grid_rf_pca, scoring='f1_macro', cv=5)

# Fitting the grid search(for cleaned dataset)
# ============================================
grid_search_rf_pca.fit(X_train_cleaned, y_train_cleaned)

# Printing the best parameters
# ============================
print(f"Best parameters: {grid_search_rf_pca.best_params_}")

# Printing the best score
# ========================
print(f"Best score: {grid_search_rf_pca.best_score_}")

# Defining the pipeline with RandomForestClassifier and lda
# =========================================================
pipe_rf_lda = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA()),
    ('classifier', RandomForestClassifier())
])

# Defining the parameters for the grid search
# ===========================================
param_grid_rf_lda = {
    'lda__n_components': [1, 2],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, 20]
}

# Defining the grid search
# ========================
grid_search_rf_lda = GridSearchCV(estimator=pipe_rf_lda, param_grid=param_grid_rf_lda, scoring='f1_macro', cv=5)

# Fitting the grid search(for cleaned dataset)
# ============================================
grid_search_rf_lda.fit(X_train_cleaned, y_train_cleaned)

# Printing the best parameters
# ============================
print(f"Best parameters: {grid_search_rf_lda.best_params_}")

# Printing the best score
# ========================
print(f"Best score: {grid_search_rf_lda.best_score_}")

# Defining the pipeline with RandomForestClassifier
# =================================================
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Defining the parameters for the grid search
# ===========================================
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, 20]
}

# Defining the grid search
# ========================
grid_search_rf = GridSearchCV(estimator=pipe_rf, param_grid=param_grid_rf, scoring='f1_macro', cv=5)

# Fitting the grid search(for cleaned dataset)
# ============================================
grid_search_rf.fit(X_train_cleaned, y_train_cleaned)

# Printing the best parameters
# ============================
print(f"Best parameters: {grid_search_rf.best_params_}")

# Printing the best score
# ========================
print(f"Best score: {grid_search_rf.best_score_}")

"""#### Confusion Matrix for the best model with 60-40 split"""

# Predicting the test data best model
# ===================================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = grid_search_svc_pca.best_estimator_
best_model.fit(X_train_c, y_train_c)
y_pred = best_model.predict(X_test_c)

# Generating the confusion matrix (Taken from the lecture)
# ========================================================
ConfusionMatrixDisplay.from_predictions(y_test_c, y_pred)
plt.title('Confusion Matrix for SVC with PCA')
plt.xticks(rotation=90)
plt.show()

"""### Final evaluation"""

# Dropping the first column from df_test
# =======================================
df_test = df_test.drop('index', axis=1)

# Converting the catagorical data to numerical data
# =================================================
df_test_dummies = pd.get_dummies(df_test, drop_first=True)

X_cleaned.info()

df_test_dummies.info()

# Fitting the best model on the training data
# ===========================================
best_model.fit(X_cleaned, y_cleaned)

# Predicting the test data
# ========================
y_pred_final = best_model.predict(df_test_dummies)

"""### Kaggle submission"""

# Mapping the diagnosis back to the original values
# =================================================
diagonosis_dict_reverse = {v: k for k, v in diagonosis_dict.items()}
y_pred_final = pd.Series(y_pred_final).map(diagonosis_dict_reverse)

#----------------------------------------
# Creating csv file for Kaggle submission
#----------------------------------------
DF = pd.DataFrame(y_pred_final)
headers = ["Diagnosis"]
DF.columns = headers
DF.to_csv('predictions.csv', index_label='index', index=True)

"""#### ROC Curve for Binary Version of the Target variable"""

# Coverting the diagoniss column to 2 classes
# ============================================
df_train_simplified = df_cleaned.copy()
df_train_simplified['Diagnosis'] = df_train_simplified['Diagnosis'].apply(lambda x: 0 if x == 0 else 1)

X_simplified = df_train_simplified.drop('Diagnosis', axis=1)
y_simplified = df_train_simplified['Diagnosis']

# Defining the classifier
# ========================
lr = LogisticRegression()
# Producing the probabilities for each class
y_prob = cross_val_predict(lr, X_simplified, y_simplified, cv=5, method='predict_proba')
fpr, tpr, thresholds = roc_curve(y_simplified, y_prob[:, 1])

# Plotting the ROC curve
# ======================
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()