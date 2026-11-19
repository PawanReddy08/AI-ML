# Customer Churn Prediction

This beginner machine-learning project predicts whether a telecom customer
is likely to churn using a Logistic Regression model.

## Dataset

The project uses the Credit card Churn dataset from Kaggle.

The dataset contains customer information such as:

- Credit Score
- Geography
- Tenure
- Gender
- Age
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- Estimated Salary
- Exited

## Objective

The goal is to predict whether a customer will churn based on their
demographic details, account information, balance, and activity .

## Project Workflow

1. Load and inspect the dataset  
2. Clean the data  
3. Perform exploratory data analysis  
4. Create basic features  
5. Split the data into training and testing sets  
6. Preprocess numeric and categorical features  
7. Train a Logistic Regression model  
8. Evaluate the model  
9. Save the trained pipeline using pickle  

## Preprocessing

- Numeric features are scaled using `StandardScaler`.
- Categorical features are encoded using `OneHotEncoder`.
- Preprocessing and the model are combined using a Scikit-learn `Pipeline`.

## Model

A Logistic Regression model was used because it is simple, interpretable,
and suitable for binary classification problems.

The model uses `class_weight="balanced"` because the churn classes are
imbalanced.

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

For churn prediction, recall is important because the business wants to
identify as many customers as possible who are likely to leave.

## Model Saving

The complete trained pipeline is saved as:

```text
churn_model.pkl
