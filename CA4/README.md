# CA4

## Rules and context of the competition

1. The general workflow is similar to that of CA3  (CA3_Workflow.pdf Download CA3_Workflow.pdf). As opposed to how things were done in CA3, however, you are now expected to use cross-validation instead of multiple train/validation splits. You can use the sci-kit learn implementations for cross-validation.

2. You have access to a range of classifiers, kernels, and L1/L2-regularization.

3. Classifier selection and hyperparameter tuning are the core of this exercise. Use the appropriate scikit-learn tools from Chapter 6 to automatically search for optimal parameter values. You are expected to make use of all the following components:

a. Hyperparameter tuning
b. Scikit-learn Pipelines
c. Cross-validation
d. Some form of dimension reduction (f.ex PCA or SFS)
e. Make a confusion matrix for your final classifier. For this, use your best model/pipeline, and perform a simple 60/40 test_train_split on the entire training dataset. Then, plot a confusion matrix from the predictions on the test.
The macro-averaged F1-score when evaluating models in your submissions. We encourage you to test out other evaluation metrics as well. 

## Background

![screenshot](https://github.com/JonyKarmakar/MachineLearning/blob/main/CA4/humanliver.png)

Liver disease is a growing global health concern, with millions affected worldwide. Early detection and accurate prediction of liver disease are crucial for timely intervention and improved patient outcomes. In this study, we aim to develop a machine learning model to predict the presence and type of liver disease based on various clinical features and biomarkers.

The dataset includes 1005 patients (703 train, 302 test) with liver disease and healthy controls. The target variable is the liver disease diagnosis, categorized into seven classes: 'Healthy', 'Cirrhosis', 'Drug-induced Liver Injury', 'Fatty Liver Disease', 'Hepatitis', 'Autoimmune Liver Diseases', and 'Liver Cancer'.

The features in the dataset encompass a wide range of clinical and laboratory parameters, such as (but not limited to): 

a. Liver function tests (ALT, AST, ALP, GGT, bilirubin, albumin) <br />
b. Complete blood count (hemoglobin, platelets, WBC, RBC) <br />
c. Metabolic markers (serum ammonia, lactate, urea, creatinine) <br />
d. Inflammatory markers (CRP, IL-6) <br />
e. Trace elements (serum copper, iron, zinc) <br />

### Data Description:

1. AFP (ng/mL): Alpha-fetoprotein, a tumor marker, measured in nanograms per milliliter.

2. ALP (U/L): Alkaline phosphatase, a liver enzyme, measured in units per liter.

3. ALT (U/L): Alanine transaminase, a liver enzyme, measured in units per liter.

4. AST (U/L): Aspartate transaminase, a liver enzyme, measured in units per liter.

5. Age: Patient's age in years.

6. Albumin (g/dL): Serum albumin, a protein produced by the liver, measured in grams per deciliter.

7. Alcohol_Use (yes/no): Indicates whether the patient consumes alcohol.

8. Bilirubin (mg/dL): Total bilirubin, a byproduct of red blood cell breakdown, measured in milligrams per deciliter.

9. CRP (mg/L): C-reactive protein, an inflammatory marker, measured in milligrams per liter.

10. Diabetes (yes/no): Indicates whether the patient has diabetes.

11. Fibroscan (kPa): Liver stiffness measurement using transient elastography, in kilopascals.

12. GGT (U/L): Gamma-glutamyl transferase, a liver enzyme, measured in units per liter.

13. Gender: Patient's gender (MALE/FEMALE).

14. Hemoglobin (g/dL): Hemoglobin concentration in the blood, measured in grams per deciliter.

15. IL-6 (pg/mL): Interleukin-6, an inflammatory cytokine, measured in picograms per milliliter.

16. Obesity (yes/no): Indicates whether the patient is obese.

17. PT/INR: Prothrombin time/international normalized ratio, a measure of blood clotting function.

18. Platelets (10^9/L): Platelet count in the blood, measured in 10^9 cells per liter.

19. RBC (10^12/L): Red blood cell count, measured in 10^12 cells per liter.

20. Serum_Ammonia (μmol/L): Serum ammonia concentration, measured in micromoles per liter.

21. Serum_Copper (μg/dL): Serum copper concentration, measured in micrograms per deciliter.

22. Serum_Creatinine (mg/dL): Serum creatinine, a marker of kidney function, measured in milligrams per deciliter.

22. Serum_Iron (μg/dL): Serum iron concentration, measured in micrograms per deciliter.

23. Serum_Lactate (mmol/L): Serum lactate concentration, measured in millimoles per liter.

24. Serum_Urea (mg/dL): Serum urea, a marker of kidney function, measured in milligrams per deciliter.

25. Serum_Zinc (μg/dL): Serum zinc concentration, measured in micrograms per deciliter.

26. TIBC (μg/dL): Total iron-binding capacity, a measure of the blood's capacity to bind iron, in micrograms per deciliter.

27. Transferrin_Saturation (%): Percentage of transferrin (an iron-binding protein) that is saturated with iron.

28. WBC (10^9/L): White blood cell count, measured in 10^9 cells per liter.

29. pH: Blood pH level, a measure of acidity or alkalinity.

Target Variable:

Liver_Disease: Categorical variable indicating the type of liver disease diagnosed ('Healthy', 'Cirrhosis', 'Drug-induced Liver Injury', 'Fatty Liver Disease', 'Hepatitis', 'Autoimmune Liver Diseases', 'Liver Cancer')



