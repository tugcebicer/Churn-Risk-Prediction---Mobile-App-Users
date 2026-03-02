# Churn Risk Prediction — Mobile App Users

A supervised machine learning project that predicts churn risk for mobile app users based on their in-app behavioral data.

---

## Project Goal

The goal of this project is to identify which users are at risk of churning (stopping using the app) based on their in-app behavior. Early churn detection allows marketing and product teams to take targeted action such as sending re-engagement campaigns, offering discounts, or improving the experience for at-risk segments.

---

## Dataset

**File:** `synthetic_userdata_app.csv`  
**Rows:** 150 users  
**Target column:** `ChurnRisk` (0 = not at risk, 1 = at risk)

| Feature | Description |
|---|---|
| `UID` | Unique user identifier |
| `AcquisitionSource` | How the user was acquired (e.g. GoogleAds, OrganicLanding) |
| `Platform` | Device platform (iOS / Android) |
| `TotalSessions` | Total number of app sessions |
| `LevelsCompleted` | Number of levels/tasks completed in-app |
| `BadgesEarned` | Number of badges/achievements earned |
| `PurchaseAmount` | Total in-app purchase amount |
| `AvgSessionLength_Min` | Average session duration in minutes |
| `RecentSessionDate` | Date of most recent session |
| `ChurnRisk` | Target label — 1 if user is at churn risk |

---

##  What Was Done & Why

### 1. Exploratory Data Analysis (EDA)
Before building any model, the data was explored to understand distributions, spot missing values, and detect outliers. This step is critical because feeding dirty data into a model leads to unreliable predictions.

### 2. Data Cleaning & Outlier Handling
Outliers were detected and handled to prevent them from skewing the model. Features like `TotalSessions` and `PurchaseAmount` can have extreme values that mislead the algorithm if left untreated.

### 3. Feature Engineering
A new feature `DaysSinceSession` was derived from `RecentSessionDate` to capture user recency — a strong signal for churn. Categorical columns (`AcquisitionSource`, `Platform`) were label-encoded for model compatibility.

### 4. Model Training
The cleaned dataset was trained using a **Gradient Boosting Classifier**, an ensemble method that builds trees sequentially to correct previous errors — making it highly effective for tabular classification tasks like churn prediction.

### 5. Model Evaluation
The model was evaluated using industry-standard metrics and visualizations: Precision-Recall curve, ROC curve, Confusion Matrix, and SHAP-based feature importance to understand what drives churn predictions.

---

##  Evaluation Results

| Metric | Score |
|---|---|
| ROC AUC | 1.000 |
| PR AUC | 1.000 |
| Macro-average F1 | 1.000 |
| Micro-average Precision | 100% |
| Micro-average Recall | 100% |
| Log Loss | ~0.05 |

### Confusion Matrix
| | Predicted 0 | Predicted 1 |
|---|---|---|
| **True 0** | 98% | 2% |
| **True 1** | 0% | 100% |

---

## Key Finding — Most Important Feature

According to SHAP-based feature attribution (Sampled Shapley method):

1. **TotalSessions** — by far the strongest predictor (~80%+ importance)
2. **PurchaseAmount**
3. **AvgSessionLength_Min**
4. **AcquisitionSource**

> Users with fewer total sessions are significantly more likely to churn. This suggests that **early engagement is the biggest retention lever** for this app.

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Model training & evaluation |
| Matplotlib | Visualizations |
| SHAP | Feature importance (Shapley values) |
| Google Colab | Notebook environment |

---

##  Repository Structure

```
├── synthetic_userdata_app.csv     # Dataset
├── churn_risk_model.ipynb         # Full notebook (EDA + Model + Evaluation)
├── eval_curves.png                # Precision-Recall, ROC, PR by threshold
├── confusion_matrix.png           # Confusion matrix
├── feature_importance.png         # SHAP feature attribution
└── README.md                      # This file
```

---

##  How to Run

1. Clone this repository
2. Open `churn_risk_model.ipynb` in [Google Colab](https://colab.research.google.com) or Jupyter
3. If using Colab, mount your Google Drive and update the dataset path:
```python
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/synthetic_userdata_app.csv')
```
4. Run all cells in order

---

## 👩‍💻 Author

**Tugce**  
Data Analyst
*This project was completed as part of a hands-on machine learning and marketing analytics portfolio.*
