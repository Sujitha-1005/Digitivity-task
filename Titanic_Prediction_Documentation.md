# 🚢 Titanic Survival Prediction — Project Report

**Dataset:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

**Files:**
- `train.csv`: Training dataset containing survival labels.
- `test.csv`: Test dataset used for generating predictions.
- `titanic_prediction.ipynb`: Jupyter notebook containing the full analysis and modeling pipeline.

---

## 1. Dataset Description

The Titanic dataset contains passenger information from the 1912 disaster. The goal is to predict whether a passenger **survived (1)** or **did not survive (0)**.

| Variable | Description | Notes |
|----------|-------------|-------|
| `survived` | Survival status | 0 = No, 1 = Yes |
| `pclass` | Ticket class | 1 = Upper, 2 = Middle, 3 = Lower |
| `sex` | Passenger sex | male / female |
| `age` | Age in years | Fractional if < 1; xx.5 if estimated |
| `sibsp` | Siblings / spouses aboard | Brother, sister, stepbrother, stepsister, husband, wife |
| `parch` | Parents / children aboard | Mother, father, daughter, son, stepdaughter, stepson |
| `ticket` | Ticket number | Unique passenger ticket identifier |
| `fare` | Passenger fare | Ticket price paid |
| `cabin` | Cabin number | Cabin number where the passenger stayed |
| `embarked` | Port of embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

**Files used:** `train.csv` (891 rows), `test.csv` (418 rows)

---

## 2. Data Cleaning

Missing values were handled as follows:

| Column | Missing % | Strategy |
|--------|-----------|----------|
| `Age` | 19.9% | Filled with **median** — robust to outliers |
| `Embarked` | 0.2% | Filled with **mode** (= 'S') |
| `Fare` | 0.24% (test only) | Filled with **median** |
| `Cabin` | ~77% | Too sparse — converted to binary `HasCabin` flag |

After cleaning: **0 missing values** in both train and test sets.

---

## 3. Feature Engineering

New features were created to improve model performance:

| Feature | Description |
|---------|-------------|
| `Title` | Extracted from Name (Mr, Mrs, Miss, Master, Rare) |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if traveling solo |
| `HasCabin` | 1 if cabin was recorded |
| `FareBand` | Fare grouped into 4 quartile bins |
| `AgeBand` | Age grouped: Child / Teen / Young Adult / Adult / Senior |

Categorical columns (`Sex`, `Embarked`, `Title`) were encoded using `LabelEncoder`.

---

## 4. Exploratory Data Analysis

Key survival patterns found in the training data:

| Factor | Insight |
|--------|---------|
| **Sex** | Female: **74.2%** survived — Male: **18.9%** survived |
| **Pclass** | 1st: **63%** — 2nd: **47%** — 3rd: **24%** |
| **Title** | Mrs: 79% — Miss: 70% — Master: 58% — Mr: 16% |
| **Family Size** | Small families (2–4) survived most; solo travelers and large groups fared worst |
| **Age** | Children (< 12) had better survival odds |
| **HasCabin** | Passengers with recorded cabin had higher survival rates |

> **Overall survival rate: 38.4%** (549 did not survive, 342 survived)

---

## 5. Visualizations

Four charts were generated:

1. **Viz 1** — Survival rate by Sex, Pclass, Embarked, and FamilySize (bar + line charts)
2. **Viz 2** — Age and Fare distributions split by survival (histograms)
3. **Viz 3** — Feature correlation heatmap (14 features)
4. **Viz 4** — Title x Pclass survival heatmap + overall survival pie chart

Additional charts: Confusion matrices, ROC curves, and Feature Importance plot.

---

## 6. Models Trained

Three models were trained using an 80/20 train-validation split with 5-Fold Cross-Validation:

| Model | Validation Accuracy | CV Accuracy (5-Fold) |
|-------|--------------------|-----------------------|
| Logistic Regression | 82.12% | 80.70% +/- 0.02 |
| Random Forest | 80.45% | 83.50% +/- 0.01 |
| **Gradient Boosting** | **83.73% (test)** | **83.61% +/- 0.03** |

> **Best model selected by CV score: Gradient Boosting**

**13 features used:** `Pclass`, `Sex_enc`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked_enc`, `HasCabin`, `FamilySize`, `IsAlone`, `Title_enc`, `FareBand`, `AgeBand`

---

## 7. Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Sex | 0.2770 |
| 2 | Fare | 0.1307 |
| 3 | Title | 0.1297 |
| 4 | Age | 0.0995 |
| 5 | Pclass | 0.0806 |
| 6 | HasCabin | 0.0724 |
| 7 | FamilySize | 0.0532 |

---

## 8. Final Predictions

The best model (Gradient Boosting) was applied to `test.csv`:
- Predictions saved to `submission.csv`

Sample output:

| PassengerId | Survived |
|-------------|----------|
| 892 | 0 |
| 893 | 0 |
| 900 | 1 |

---

## 9. How to Run
1. Ensure all dependencies (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) are installed.
2. Place `train.csv` and `test.csv` in the same directory as `titanic_prediction.ipynb`.
3. Execute the cells in the notebook to reproduce the results and generate `submission.csv`.

---

## 10. Conclusion

The **"women and children first"** protocol is clearly reflected in the data — **gender** and **socioeconomic status** (Pclass, Fare) were the strongest predictors of survival. Gradient Boosting achieved the best generalisation at **~83.7% accuracy**. Engineered features like `Title`, `FamilySize`, and `HasCabin` meaningfully improved predictions beyond the raw columns alone.

---

*Tools used: Python · Pandas · Matplotlib · Seaborn · Scikit-learn*
