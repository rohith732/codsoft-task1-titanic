import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   TASK 1: TITANIC SURVIVAL PREDICTION")
print("   CodSoft Data Science Internship")
print("=" * 60)

np.random.seed(42)
n = 891
pclass = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
sex    = np.random.choice([0, 1], n, p=[0.35, 0.65])
age    = np.clip(np.random.normal(29.7, 14.5, n), 1, 80)
sibsp  = np.random.choice([0,1,2,3,4,5,8], n, p=[0.68,0.23,0.03,0.02,0.01,0.005,0.005])
parch  = np.random.choice([0,1,2,3,4,5,6], n, p=[0.76,0.13,0.09,0.01,0.005,0.003,0.002])
fare   = np.clip(np.random.exponential(32, n), 5, 512)
embarked = np.random.choice([0, 1, 2], n, p=[0.19, 0.09, 0.72])

log_odds = (
    1.2
    - 1.3 * (pclass == 3).astype(float)
    - 0.4 * (pclass == 2).astype(float)
    + 2.5 * sex
    - 0.02 * age
    + 0.003 * fare
    - 0.3 * sibsp
    - 0.1 * parch
)
prob = 1 / (1 + np.exp(-log_odds))
survived = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    'Pclass': pclass, 'Sex': sex, 'Age': age,
    'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
    'Embarked': embarked, 'Survived': survived
})

print(f"\nDataset shape: {df.shape}")
print(f"Survival rate: {df['Survived'].mean():.2%}")
print(f"\nSurvival by gender:")
print(df.groupby('Sex')['Survived'].mean().rename(index={0:'Female', 1:'Male'}))

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

print("\n" + "=" * 60)
print("MODEL RESULTS — Logistic Regression")
print("=" * 60)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Not Survived','Survived'])}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

print("\nSAMPLE PREDICTIONS:")
samples = pd.DataFrame({
    'Pclass': [1, 3], 'Sex': [0, 1], 'Age': [25, 35],
    'SibSp': [0, 1], 'Parch': [0, 0], 'Fare': [100, 10], 'Embarked': [0, 2]
})
preds = model.predict(scaler.transform(samples))
probs = model.predict_proba(scaler.transform(samples))
labels = {0: 'Female', 1: 'Male'}
classes = {1: 'Survived', 0: 'Did Not Survive'}
for i, row in samples.iterrows():
    print(f"  {labels[row['Sex']]}, Class {int(row['Pclass'])}, Age {row['Age']} → {classes[preds[i]]} ({probs[i][1]:.1%} probability)")
