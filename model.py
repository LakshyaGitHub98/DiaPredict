import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load Dataset
df = pd.read_csv("diabetes.csv")

# 2️⃣ Basic Info
print(df.head())
print(df.isnull().sum())

# 3️⃣ Replace 0 values with median (important cleaning step)
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

# 4️⃣ Split Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7️⃣ Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 8️⃣ Prediction
y_pred = model.predict(X_test)

# 9️⃣ Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

# 🔟 Save Model & Scaler
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))