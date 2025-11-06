# 🏡 House Price Prediction

This project predicts house prices using **Linear Regression** and **Random Forest Regressor** models.  
It is based on the California Housing dataset and demonstrates basic regression model training and evaluation in Python.

---
## 📊 Dataset

The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/).
You can find it here: [California Housing Prices Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

The dataset contains information about various housing features in California, such as median income, house age, and proximity to the ocean, which are used to predict median house values.


## 📊 Models and Results

| Model | R² Score |
|-------|-----------|
| Linear Regression | 0.6156 |
| Random Forest Regressor | 0.9747 (train set) |

> ⚠️ The Random Forest model shows higher accuracy but may be overfitting, so cross-validation or parameter tuning (e.g., GridSearchCV) is recommended.

---

## 🧰 Tools & Libraries

- Python  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/arzunajafzade/house_price_prediction.git
