# CA5

## Rules and context of the competition

![screenshot](https://github.com/JonyKarmakar/MachineLearning/blob/main/CA5/fruits.png)

A food company wants to automate sorting chili peppers according to spiciness level to distribute the peppers to the best market. They have a variety of measurements on the chili peppers (see dataset description). You will help them to train a machine learning model for that task.

1. The general workflow is similar to that of the previous Kaggle competitions, although regression is now the focus of the assignment.

2. You have access to a wide range of models and preprocessing tools taught in lectures. All models presented in the lecture are allowed. Models that were not presented in the lecture are not allowed. Use the appropriate scikit-learn tools to preprocess and automatically search for optimal parameter values.

3. You may try to engineer new features if you wish to do so. If you are unsure how to structure your notebook follow the steps: Imports, Reading the data, Data exploration and visualization, Data cleaning, Data exploration and visualization after cleaning, Data preprocessing (and possibly some more visualization), Evaluating different models/hyperparameters (using cross-validation), Kaggle submission (with prediction and result on the test set).

4. Bonus task (not mandatory): Measuring features is expensive. The company would like to measure as few features as possible. What is the minimum amount of original features (and which features) you can use without sacrificing too much performance? The company is willing to accept an MAE of 10% less than the MAE of your best model.

5. Your submission must use the following components/techniques:

Hyperparameter tuning
Scikit-learn Pipelines
Cross-validation
The Mean Absolute Error (MAE) as the performance metric (this is what is scored on Kaggle).
Moreover, you must build and evaluate at least two of the following three model pipelines (A-C). All pipelines should contain suitable preprocessing. At least one of your pipelines has to use a model based on linear regression (one of the linear regression-based regressors covered in the lecture, e.g. multi-linear regression/polynomial regression/PCR/PLS/RANSAC/regularized regression...).

(A) Regression analysis.

(B) Multi-class classification analysis with an ensemble classifier. To this end bin the target variable into discrete chunks and train a classification model on the binned target variable. For binning continuous data (transforming continuous target to discrete target values), you can use pandas.qcut() or pandas.cut() (for example code look at the end of this task description). Test and comment on how the number of bins affects the prediction.

(C) A two-step analysis (two sequential pipelines). In the first step, train an ensemble classification model to separate bell peppers (peppers with a SHU of 0) and spicy peppers (SHU larger than 0). In the second step, use a regression model to estimate the scoville score (SHU) of those samples that the binary classifier identifies as spicy peppers. Combine the results of both steps into a single prediction vector.

## Data Description:

1. Length (cm): The length of the pepper measured in centimeters.
2. Width (cm): The width of the pepper measured in centimeters.
3. Weight (g): The weight of the pepper measured in grams.
4. Pericarp Thickness (mm): The thickness of the pepper's outer wall (pericarp) measured in millimeters.
5. Seed Count: The number of seeds found inside the pepper.
6. Capsaicin Content: The amount of capsaicin, the compound responsible for the pepper's spiciness, present in the pepper.
7. Vitamin C Content (mg): The amount of Vitamin C, measured in milligrams, found in the pepper.
8. Sugar Content: The amount of sugar present in the pepper.
9. Moisture Content: The percentage of moisture content in the pepper.
10. Firmness: A measure of the pepper's firmness.
11. Color: The color of the pepper (e.g., red, yellow)
12. Harvest Time: The time of day when the pepper was harvested (e.g., Midday).
13. Average Daily Temperature During Growth (Celsius): The average daily temperature, measured in Celsius, during the pepper's growth period.
14. Average Temperature During Storage (celsius): The average temperature during the storage of the pepper, post-harvest.
15. Scoville Heat Units (SHU): A measure of the pepper's spiciness using the Scoville scale. (TARGET)
