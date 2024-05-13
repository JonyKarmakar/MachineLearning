"""
# CA3 - Lumifruit Edibility Classification
### Jony Karmakar

### Imports
"""

# Importing Libaries
# ======================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# Checking the info
df_train.info()

df_train.describe()

"""#### Histogram"""

# Plotting histograms for all columns in the dataset
df_train.hist(bins=10, figsize=(26, 15))
plt.show()

"""#### Box Plots"""

# Removing the target variable
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
features = df_train.columns[:-1]

plt.figure(figsize=(20, 60))

for i, column in enumerate(features):
    plt.subplot(len(features), 4, i + 1)
    sns.violinplot(y=df_train[column])
    plt.title(column)

plt.tight_layout()
plt.show()

"""#### Heatmap"""

corr_matrix = df_train.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cbar=True)

plt.show()

"""#### Scatter Plot"""

sns.pairplot(df_train, hue="Edible")
plt.show()

"""### Data cleaning

#### Handling Missing Values
"""

# Impute missing values using the column mean (Taken from Lecture)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # other popular choices: "median", "most_frequent"
imputer.fit(df_train.values)
imputed_data = imputer.transform(df_train.values)

# show the dataset
# note that the output of the SimpleImputer is a NumPy array
# so we need to convert it back to a pandas DataFrame to use our helper function
df_train_imp = pd.DataFrame(imputed_data, columns=df_train.columns)
df_train_imp.index = df_train.index

df_train_imp.isnull().sum()

"""#### Detecting the Outliers"""

# Detect the outliers using the z-score
outliers_count = {}

# Iterating through each column in df_train_dna
for column in df_train_imp.columns:
    samples = df_train_imp[column].values
    z_scores = (samples - np.mean(samples)) / np.std(samples)
    outliers = np.abs(z_scores) > 3
    outliers_count[column] = np.sum(outliers)

for feature, count in outliers_count.items():
    print(f"Number of outliers in {feature}: {count}")

"""##### For this assignment outliers have been ignored.

### Data preprocessing and visualisation

#### Splitting the Dataset
"""

#-----------------------------------------
# Spliting the dataset into test and train
#-----------------------------------------
X_imp = df_train_imp.iloc[:, :-1].values
y_imp = df_train_imp.iloc[:, -1].values
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_imp, y_imp, test_size=0.20, stratify=y_imp, random_state=42)

"""##### Scaling the Dataset"""

# =======================================================================================================
# Standardising unsing the StandardScaler (Taken from the lecture also this is the common way of scaling)
# =======================================================================================================
# Initialise standard scaler and compute mean and stddev from training data
sc = StandardScaler()
sc.fit(X_train_imp)

# Transform (standardise) both X_train_imp and X_test_imp with mean and stddev from
# training data
X_train_imp_sc = sc.transform(X_train_imp)
X_test_imp_sc = sc.transform(X_test_imp)

"""#### Exploring training data after cleaning and scaling"""

# Converting numpy array to dataframe
X_train_imp_sc_df = pd.DataFrame(X_train_imp_sc)
X_test_imp_sc_df = pd.DataFrame(X_test_imp_sc)

X_train_imp_sc_df.info()

X_train_imp_sc_df.describe()

# Plotting histograms for all columns in the dataset
X_train_imp_sc_df.hist(bins=10, figsize=(26, 15))
plt.show()

"""#### Box Plots"""

features = X_train_imp_sc_df.columns

plt.figure(figsize=(20, 60))

for i, column in enumerate(features):
    plt.subplot(len(features), 4, i + 1)
    sns.boxplot(y=X_train_imp_sc_df[column])
    plt.title(column)

plt.tight_layout()
plt.show()

"""#### Violin Plots"""

features = X_train_imp_sc_df.columns

plt.figure(figsize=(20, 60))

for i, column in enumerate(features):
    plt.subplot(len(features), 4, i + 1)
    sns.violinplot(y=X_train_imp_sc_df[column])
    plt.title(column)

plt.tight_layout()
plt.show()

"""### Modelling

##### Random Forest
"""

#------------------------------------------------------------------------------------
# RANDOM FOREST()
# Parameters tuned for this model
# 'n_estimators' : Number of Trees(High->Robustness but time consuming)
# 'max_depth' : Max depth of Trees(High->Accurate model but overfitting)
# 'max_features' : The number of features to consider when looking for the best split
#------------------------------------------------------------------------------------
# Parameters to test
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 25], 'max_features': [*np.arange(0.1, 1.1, 0.1)]}

# Store the best parameter combination and its accuracy
best_params_rf = None
best_accuracy_rf = 0

for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for max_feature in param_grid['max_features']:
            accuracies = []
            for random_state in range(40, 50):
                # Splitting Dataset
                X_train, X_val, y_train, y_val = train_test_split(X_train_imp, y_train_imp, test_size=0.25, random_state=random_state)
                # Standardize the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                #  Defining the model
                rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_feature, random_state=42)
                # Fitting the model
                rfc.fit(X_train_scaled, y_train)
                # Evaluating on the test dataset
                y_pred = rfc.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, y_pred)
                accuracies.append(accuracy)

            # Measuring the average accuracy for different splits of dataset
            accuracy = np.mean(accuracies)

        if accuracy > best_accuracy_rf:
            best_accuracy_rf = accuracy
            best_params_rf= {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_feature}

print("Best Parameters:", best_params_rf)
print("Best Accuracy:", best_accuracy_rf)

"""##### Testing for X_train and X_test of Train_df"""

rfc = RandomForestClassifier(n_estimators=best_params_rf['n_estimators'], max_depth=best_params_rf['max_depth'], max_features=best_params_rf['max_features'], random_state=42)
rfc.fit(X_train_imp_sc, y_train_imp)

y_pred_rfc = rfc.predict(X_test_imp_sc)
accuracy_rfc = accuracy_score(y_pred_rfc, y_test_imp)
print(f'For Random Forest with Best Parameters: {best_params_rf} and Accuracy: {accuracy_rfc}')

"""##### Random Forest
For Random Forest we can see it has the best accuracy

### Final evaluation
"""

#------------------------------------------------
# Train the best model on the entire training set
#------------------------------------------------
scaler = StandardScaler()
scaler.fit(X_imp)
X_train_scaled = scaler.transform(X_imp)
rfc.fit(X_train_scaled, y_imp)

#-------------------------------------------------------
# Use the test set to evaluate the best k on unseen data
#-------------------------------------------------------
X_test_scaled = scaler.transform(df_test)
y_pred_rf = rfc.predict(X_test_scaled)

"""### Kaggle submission"""

#----------------------------------------
# Creating csv file for Kaggle submission
#----------------------------------------
DF = pd.DataFrame(y_pred_rf)
headers = ["Edible"]
DF.columns = headers
DF.to_csv('predictions.csv', index_label='index', index=True)