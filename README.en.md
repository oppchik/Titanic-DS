<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Language-Русский_🇷🇺-gray?style=flat-square" alt="RU"></a>
  <img src="https://img.shields.io/badge/Language-English_🇬🇧-brightgreen?style=flat-square" alt="EN">
</p>

---

# Titanic Passenger Survival Analysis & Machine Learning Modeling

This repository contains a student project completed for the "Introduction to Engineering" course (2nd semester). The goal of the project is to perform exploratory data analysis (EDA), identify patterns using a correlation matrix, and train a machine learning model to predict passenger survival.

![Titanic](/Titanic-model/titanic_photo.jpg)

## 🛠️ Tech Stack

- **Development Environment:** Jupyter Notebook
- **Programming Language:** Python
- **Key Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## 📂 Repository Structure
```
├── titanic.csv                 # Raw dataset
├── titanic_correlation.ipynb   # Data preprocessing + correlation matrix + plots
├── titanic_correlation.png     # Correlation matrix plot
├── titanic_analysis.png        # Visual data analysis plots
├── titanic_ml_model.ipynb      # ML model training + evaluation
├── confusion_matrix.png        # Confusion matrix plot
├── roc_curve.png               # ROC curve plot
├── metrics_heatmap.png         # Classification metrics heatmap
└── metrics_bar.png             # Precision / Recall / F1 comparison bar chart
```
---

## 📊 Part 1. Correlation Matrix

### Data Preprocessing

Before building the correlation matrix, the data underwent the following preparation steps:
- Removed non-numeric columns (`Name`, `Ticket`, `Cabin`).
- Encoded gender and port of embarkation numerically (`male -> 0`, `female -> 1`).
- Imputed missing values in the age column using the median.
- Created engineered features: `FamilySize = SibSp + Parch` and `IsAlone`.

### Heatmap

![Correlation Matrix](/Titanic-model/titanic_correlation.png)

The correlation matrix illustrates the linear relationship between each pair of features. Values range from **−1** (blue, strong negative correlation) to **+1** (red, strong positive correlation).

### Key Correlations with Survival

| Feature | Correlation | Interpretation |
|---|---|---|
| **Sex** | +0.54 | Gender is the strongest factor. Females survived in 74% of cases compared to 19% for males. |
| **Pclass** | −0.34 | 1st class passengers survived three times more often than 3rd class passengers. |
| **Fare** | +0.26 | Ticket price correlates positively with class status and survival rates. |
| **IsAlone** | −0.20 | Solo travelers survived less frequently; navigating to lifeboats was harder without family support. |
| **Age** | −0.06 | Shows almost no linear correlation with survival. |

> **Conclusion:** The correlation matrix highlights linear dependencies. Gender and passenger class stand out as the primary survival factors.

---

## 🤖 Part 2. Machine Learning Model (ML)

### Methodology

Two models were trained using the following features: `Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone, Title`.  
The data was split into **80% for training** and **20% for testing** (utilizing `stratify` to preserve class proportions).

Preprocessing was implemented via a `Pipeline` combined with a `ColumnTransformer`:
- **Numerical features:** Median imputation followed by `StandardScaler`.
- **Categorical features:** Mode imputation followed by `OneHotEncoder`.

### Models

> Note: The MLP model was trained to serve as a benchmark against the GBM model.
 
| Model | Description |
|---|---|
| **Gradient Boosting** | An ensemble of decision trees, where each successive tree corrects the errors of the previous ones. |
| **MLP (Neural Network)** | Two hidden layers (100 + 50 neurons), ReLU activation, and Adam optimizer. |

### Results

| Metric | Gradient Boosting |
|---|---|
| **Accuracy** | **82.7%** |
| **AUC** | **0.837** |

### Confusion Matrix

![Confusion Matrix](/Titanic-model/confusion_matrix.png)

- **99** passengers were correctly classified as deceased (True Negatives - TN)
- **49** passengers were correctly classified as survivors (True Positives - TP)
- **11** deceased passengers were incorrectly predicted as survivors (False Positives - FP)
- **20** survivors were incorrectly predicted as deceased (False Negatives - FN)

### ROC Curve

![ROC Curve](/Titanic-model/roc_curve.png)

An **AUC of 0.837** indicates that there is an 83.7% chance that the model will rank a randomly chosen surviving passenger higher than a randomly chosen deceased passenger. For comparison, a random guess yields an AUC of 0.5.

### Metrics by Class

![Classification Metrics Heatmap](/Titanic-model/metrics_heatmap.png)

![Precision, Recall, F1 Comparison](/Titanic-model/metrics_bar.png)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| **Did Not Survive (0)** | 0.832 | 0.900 | 0.865 |
| **Survived (1)** | 0.817 | 0.710 | 0.760 |

- **Precision (Did Not Survive) = 0.832:** When predicting a passenger's demise, the model is correct 83% of the time.
- **Recall (Survived) = 0.710:** The model correctly identifies 71% of all actual survivors.
- The model performs better at identifying deceased passengers than survivors. This is primarily due to class imbalance in the dataset (62% deceased vs. 38% survived).

---

## 📌 Summary Conclusions

Both the data analysis and the machine learning model validate well-known historical facts about the shipwreck:

1. **Gender** was the defining factor: women survived 3.9 times more often than men.
2. **Passenger Class**: 1st class passengers received priority access to the lifeboats.
3. **Social Status** (extracted via the `Title` feature) proved to be one of the strongest predictors for the model.
4. **Solo travelers** survived less frequently than passengers traveling with small families.

The **Gradient Boosting** model achieves an accuracy of **82.7%** and an AUC of **0.837**, comfortably outperforming baseline models and demonstrating stable results under cross-validation.
