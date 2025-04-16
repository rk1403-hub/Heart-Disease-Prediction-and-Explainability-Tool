import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import joblib

file_path = '/Users/prajwalrk/Desktop/mini_project/heart_disease_uci.csv'
df = pd.read_csv(file_path)

df_clean = df.drop(['id', 'dataset'], axis=1)
df_clean = df_clean.dropna()
df_clean = pd.get_dummies(df_clean, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
df_clean['sex'] = df_clean['sex'].apply(lambda x: 1 if x == 'Male' else 0)
df_clean['fbs'] = df_clean['fbs'].apply(lambda x: 1 if x else 0)
df_clean['exang'] = df_clean['exang'].apply(lambda x: 1 if x else 0)

X = df_clean.drop('num', axis=1)
y = df_clean['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3)
grid_search.fit(X_train_res, y_train_res)

print(f"Best parameters found: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_

y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))

roc_auc_rf = roc_auc_score(y_test, best_rf_model.predict_proba(X_test), multi_class='ovr')
print(f"Random Forest ROC-AUC Score: {roc_auc_rf}")

xg_model = xgb.XGBClassifier(random_state=42)
xg_model.fit(X_train_res, y_train_res)
y_pred_xg = xg_model.predict(X_test)

print("XGBoost Classifier Report:")
print(classification_report(y_test, y_pred_xg))

roc_auc_xg = roc_auc_score(y_test, xg_model.predict_proba(X_test), multi_class='ovr')
print(f"XGBoost ROC-AUC Score: {roc_auc_xg}")

feature_importances = best_rf_model.feature_importances_
print("Feature Importances (Random Forest):")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

explainer_lime = LimeTabularExplainer(
    X_train.values,
    training_labels=y_train.values,
    mode='classification',
    feature_names=X.columns,
    class_names=['No Heart Disease', 'Heart Disease'],
    kernel_width=3,
    discretize_continuous=True
)

def predict_new_user():
    print("\n----- Heart Disease Prediction for New User -----")
    age = float(input("Age: "))
    sex = int(input("Sex (1 for Male, 0 for Female): "))
    cp = int(input("Chest pain type (1, 2, 3, or 4): "))
    trestbps = float(input("Resting blood pressure: "))
    chol = float(input("Cholesterol level: "))
    thalch = float(input("Maximum heart rate achieved (thalch): "))
    fbs = int(input("Fasting blood sugar >120 mg/dl (1 for Yes, 0 for No): "))
    exang = int(input("Exercise induced angina (1 for Yes, 0 for No): "))
    oldpeak = float(input("ST depression induced by exercise: "))
    slope = int(input("Slope of the peak exercise ST segment (1, 2, or 3): "))
    ca = int(input("Number of major vessels colored by fluoroscopy (0-3): "))
    thal = int(input("Thalassemia (3=normal, 6=fixed, 7=reversible): "))

    user_row = {
        'age': age,
        'sex': sex,
        'trestbps': trestbps,
        'chol': chol,
        'thalch': thalch,
        'fbs': fbs,
        'exang': exang,
        'oldpeak': oldpeak,
        'ca': ca,
        'cp_atypical angina': 1 if cp == 2 else 0,
        'cp_non-anginal': 1 if cp == 3 else 0,
        'cp_typical angina': 1 if cp == 4 else 0,
        'restecg_normal': 0,
        'restecg_st-t abnormality': 0,
        'slope_flat': 1 if slope == 2 else 0,
        'slope_upsloping': 1 if slope == 3 else 0,
        'thal_normal': 1 if thal == 3 else 0,
        'thal_reversable defect': 1 if thal == 7 else 0
    }

    user_input = pd.DataFrame([user_row])
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    prediction = best_rf_model.predict(user_input)
    if prediction[0] == 1:
        print("\nPrediction: Heart Disease Risk")
    else:
        print("\nPrediction: No Heart Disease Risk")

    exp = explainer_lime.explain_instance(user_input.values[0], best_rf_model.predict_proba)
    exp.as_pyplot_figure()
    plt.show()

predict_new_user()

print("\nGenerating Custom Partial Dependence Plot for 'age' feature...")
def compute_partial_dependence(model, X, feature, grid_resolution=20):
    feature_values = np.linspace(X[feature].min(), X[feature].max(), grid_resolution)
    predictions = []
    for idx, value in enumerate(feature_values):
        X_copy = X.copy()
        X_copy[feature] = value
        pred = model.predict_proba(X_copy)[:, 1]
        predictions.append(np.mean(pred))
        print(f"{idx+1}/{grid_resolution} completed")
    return feature_values, predictions

feature_values, pdp_predictions = compute_partial_dependence(best_rf_model, X_train.sample(200, random_state=42), 'age')
plt.figure(figsize=(8, 6))
plt.plot(feature_values, pdp_predictions, label='Partial Dependence of Age')
plt.xlabel('Age')
plt.ylabel('Avg. Prediction Probability')
plt.title('Partial Dependence of Age on Heart Disease Risk')
plt.legend()
plt.show()

joblib.dump(best_rf_model, 'heart_disease_rf_model.pkl')
print("Model saved as 'heart_disease_rf_model.pkl'.")
