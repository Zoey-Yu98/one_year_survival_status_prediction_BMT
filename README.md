Thank you for sharing the link to your repository! Now that I have access to the repository, here's a refined README template that you can customize for your project based on the structure and code you’ve shared:

---

# One Year Survival Status Prediction for Pediatric Bone Marrow Transplant Patients

**Description:**  
This project involves predicting the one-year survival status (alive or deceased) of pediatric patients who underwent Bone Marrow Transplantation (BMT). The model builds on various machine learning algorithms such as Random Forest, Logistic Regression, SVC, and XGBoost to handle classification tasks, with additional techniques for dealing with class imbalance, missing data, and hyperparameter optimization.

---

## **Table of Contents**
1. [Installation](#installation)
2. [Environment Setup](#environment-setup)
3. [Usage](#usage)
4. [Data](#data)
5. [Models and Evaluation](#models-and-evaluation)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Installation**

To get started with the project, clone this repository and install the necessary dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/Zoey-Yu98/one_year_survival_status_prediction_BMT.git
   ```
2. Navigate to the project directory:
   ```bash
   cd one_year_survival_status_prediction_BMT
   ```
   
---

## **Environment Setup**

To create a clean environment, follow these steps:

### 1. **Create a Virtual Environment**
```bash
python -m venv venv
```

### 2. **Activate the Virtual Environment**
- On **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. **Install Dependencies**
Install the required dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

Or manually install dependencies with:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap
```

### 4. **Create the `requirements.txt` (Optional)**
If you make any updates to the environment, generate a `requirements.txt` to freeze the current dependencies:
```bash
pip freeze > requirements.txt
```

---

## **Usage**

1. **Data Preprocessing**:  
   - The dataset is loaded and processed with necessary transformations and imputations to handle missing values and class imbalance.
   
2. **Modeling**:  
   - The project explores several classifiers: Random Forest, Logistic Regression, SVC, XGBoost, and KNN.
   - **GridSearchCV** and **Cross-validation** techniques are applied for hyperparameter tuning and evaluation.

3. **Evaluation**:  
   - Models are evaluated using accuracy, F1-score, confusion matrix, and SHAP for interpretability.

### Example to run the Random Forest model:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
X = # your features here
y = # your target variable here

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## **Data**

The dataset used in this project contains features and outcomes of pediatric BMT patients. The data includes various features such as patient demographics, medical history, and transplant specifics. The goal is to predict whether the patient will survive for one year post-transplant.

---

## **Models and Evaluation**

### **Models Used:**
- **Random Forest Classifier**: Handles class imbalance and provides feature importance.
- **Logistic Regression**: Simple and interpretable, used as a baseline model.
- **SVC**: Support vector classifier for handling high-dimensional spaces.
- **XGBoost**: Gradient boosting for high-performance model.
- **KNN**: For comparison with distance-based classifiers.

### **Evaluation Metrics:**
- **Accuracy Score**: Measures the percentage of correctly classified instances.
- **F1-Score**: Balances precision and recall, useful for imbalanced datasets.
- **Confusion Matrix**: Evaluates model’s classification performance.
- **SHAP**: Provides explainability and insights into model behavior.

---

## **Results**

Here you can present key results, such as:

- Best performing model: XGBoost with hyperparameter tuning
- Sample **Confusion Matrix** and **F1-Score** results

Example SHAP visualization for feature importance:
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
```

---

## **Contributing**

Feel free to fork the repository, submit pull requests, or suggest improvements. If you want to contribute, please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Push to your fork (`git push origin feature-branch`)
5. Open a Pull Request

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
