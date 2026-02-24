# DiaPredict – AI-Based Diabetes Risk Prediction System

DiaPredict is a Machine Learning-based system designed to predict the risk of diabetes using medical health parameters.  
This project is built as a hands-on learning initiative under mentor guidance, focusing on practical implementation of ML concepts.

---

## 📌 Project Objective

To develop a supervised machine learning model that:

- Predicts whether a person is diabetic or not
- Provides probability-based risk estimation
- Demonstrates end-to-end ML workflow implementation

---

## 🧠 Machine Learning Approach

- Type: Supervised Learning (Classification)
- Algorithm Used: Logistic Regression
- Train-Test Split: 80/20
- Feature Scaling: StandardScaler
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

Initial Model Accuracy: **~80%**

Special Focus:
High recall for diabetic cases to reduce false negatives in medical prediction.

---

## 📊 Dataset Used

- Diabetes dataset (Pima-based structured dataset)
- Features:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- Target:
  - 0 → Non-Diabetic
  - 1 → Diabetic

---

## ⚙️ Project Workflow

1. Data Loading
2. Data Cleaning (Replacing zero values with median)
3. Train-Test Split
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. Model Saving using Pickle


## 🚀 How to Run the Project

### 1️⃣ Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```

### 2️⃣ Run Model Training
    python model.py

The model will:

. Train
. Print accuracy and classification report
. Save the trained model as .pkl files

files

### 🔮 Future Improvements

. Implement Random Forest / Ensemble Models

. Add Confusion Matrix Visualization

. Build Flask API for model deployment

. Integrate with Node.js backend

. Connect to Flutter frontend application

. Deploy on cloud platform


### 🎯 Learning Outcome

This project helped in understanding:

. Supervised Classification

. Model Evaluation Metrics

. Data Preprocessing

. Feature Scaling

. Model Serialization

. Practical ML Implementation


### 👨‍💻 Author

Lakshya Tripathi
https://www.linkedin.com/in/lakshya-tripathi-3b205b295/
BCA Student | Aspiring Full-Stack & AI Developer

### 📌 Status

🟢 Phase 1 Complete – Model Training
🟡 Phase 2 – API Integration (In Progress)
