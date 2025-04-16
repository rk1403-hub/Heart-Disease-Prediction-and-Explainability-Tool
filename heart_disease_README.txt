# Heart Disease Prediction and Explainability Tool

This project uses machine learning to predict the likelihood of heart disease based on patient health records. It integrates model training, evaluation, and explainability features using LIME and Partial Dependence Plots.

---

## 📂 Input Data

The dataset used is:
/Users/prajwalrk/Desktop/mini_project/heart_disease_uci.csv

### Required Columns:
- age: Age of the patient
- sex: Gender (1 = Male, 0 = Female)
- cp: Chest pain type (1, 2, 3, or 4)
- trestbps: Resting blood pressure
- chol: Cholesterol level
- thalch: Maximum heart rate achieved
- fbs: Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No)
- exang: Exercise-induced angina (1 = Yes, 0 = No)
- oldpeak: ST depression induced by exercise
- slope: Slope of the peak exercise ST segment (1, 2, or 3)
- ca: Number of major vessels coloured by fluoroscopy (0–3)
- thal: Thalassemia (3 = normal, 6 = fixed, 7 = reversible)

---

## ⚙️ Workflow

### 1. Preprocessing
- Drops irrelevant columns (id, dataset)
- Cleans and one-hot encodes categorical data
- Converts binary columns to numerical format

### 2. Data Handling
- Splits data into training (70%) and testing (30%)
- Uses SMOTE to balance the training data

### 3. Model Training
- Trains a Random Forest with GridSearchCV for hyperparameter tuning
- Trains an XGBoost classifier

### 4. Model Evaluation
- Classification report and ROC-AUC score for both models
- Feature importance plot for the Random Forest model

### 5. Explainability
- LIME used to interpret predictions for custom user input
- Partial Dependence Plot (PDP) created for the age feature

### 6. Interactive Prediction
- predict_new_user() function collects input, predicts, and explains the result using LIME

### 7. Model Export
- The trained Random Forest model is saved as heart_disease_rf_model.pkl

---

## User Input Guide (Short Description)

When running the script, you'll be asked to input:
- Age: (e.g., 54)
- Sex: 1 for Male, 0 for Female
- Chest pain type: 1–4
- Resting blood pressure: e.g., 130
- Cholesterol: e.g., 250
- Max heart rate: e.g., 150
- Fasting blood sugar > 120: 1 = Yes, 0 = No
- Exercise induced angina: 1 = Yes, 0 = No
- ST depression (oldpeak): e.g., 2.3
- ST slope: 1–3
- Number of major vessels: 0–3
- Thalassemia: 3, 6, or 7

### What Each Input Means

| Field                          | Meaning |
|--------------------------------|---------|
| Age                            | Age of the patient in years |
| Sex                            | 1 = Male, 0 = Female |
| Chest pain type (cp)           | 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic |
| Resting blood pressure         | In mm Hg; higher values may indicate hypertension |
| Cholesterol                    | Serum cholesterol in mg/dl |
| Maximum heart rate achieved    | During stress test |
| Fasting blood sugar (fbs)      | 1 if > 120 mg/dl, else 0 |
| Exercise-induced angina        | 1 = Yes, 0 = No |
| Oldpeak                        | ST depression during exercise |
| Slope                          | 1 = Upsloping, 2 = Flat, 3 = Downsloping |
| ca                             | Number of major vessels observed (0–3) |
| thal                           | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect |

---

##  Output
- **Text**: Classification reports, ROC-AUC scores
- **Plots**: Feature importances, LIME explanation, PDP for age
- **File**: heart_disease_rf_model.pkl

---

## Dependencies
Install with:
pip install pandas numpy scikit-learn imbalanced-learn xgboost lime matplotlib joblib

---

## Notes
- If you'd like to add sample inputs or automate predictions from a CSV or JSON file, feel free to extend the predict_new_user() function.
- Sample entries and batch predictions can be added for demonstration if needed.
