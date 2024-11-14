
# Telecom Customer Churn Prediction

This repository contains the project for predicting customer churn in the telecom industry. The goal of the project is to explore customer data, perform clustering, implement machine learning algorithms, and ultimately predict whether a customer will churn based on various features. The dataset is sourced from Kaggle and includes details about customer behavior, demographics, and service usage.

## Dataset

The dataset used in this project is from Kaggle, which contains customer information from a telecom company. You can download the dataset from the following link:

[Kaggle Telecom Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Features in the dataset:
- **CustomerID**: Unique ID for each customer
- **Gender**: Gender of the customer
- **SeniorCitizen**: Whether the customer is a senior citizen (1 or 0)
- **Partner**: Whether the customer has a partner (Yes or No)
- **Dependents**: Whether the customer has dependents (Yes or No)
- **Tenure**: The number of months the customer has been with the company
- **PhoneService**: Whether the customer has phone service (Yes or No)
- **MultipleLines**: Whether the customer has multiple lines (Yes, No, or No phone service)
- **InternetService**: The type of internet service the customer has (DSL, Fiber optic, or No)
- **OnlineSecurity**: Whether the customer has online security (Yes or No)
- **PaperlessBilling**: Whether the customer uses paperless billing (Yes or No)
- **PaymentMethod**: The payment method used by the customer
- **MonthlyCharges**: The monthly charges for the customer
- **TotalCharges**: The total charges for the customer
- **Churn**: Whether the customer churned (Yes or No)

---

## File 1: EDA and K-Means Clustering

In this notebook, we perform Exploratory Data Analysis (EDA) to better understand the dataset and use K-means clustering to group customers into three clusters based on their characteristics. This helps in identifying different customer segments that exhibit similar behavior, such as high spenders, long-term customers, and likely churners.

### Key Tasks:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- K-means clustering to segment customers into 3 groups
- Visualization of the clusters to understand customer characteristics

You can access the EDA and clustering notebook here:  
[EDA and K-Means Clustering Notebook](https://github.com/dangwthao/Telecom-Customers-Churn-Prediction/blob/main/Telco%20customer%20churn%20-%20EDA%20%26%20Clustering.ipynb)

---

## File 2: Machine Learning Algorithms Implementation

In this notebook, we implement machine learning algorithms to predict customer churn. The process includes feature engineering, training multiple algorithms, and selecting the best model based on performance.

### Key Tasks:
- **Feature Engineering**: Creation of new features to improve model performance
- **K-Fold Cross Validation and AUC ROC**: We use K-fold cross-validation to evaluate models and select the top two algorithms based on the AUC ROC score
- **Hyperparameter Tuning**: We tune hyperparameters of the Logistic Regression and Random Forest Classifier using GridSearchCV (for Logistic Regression) and RandomizedSearchCV (for Random Forest)
- **Model Evaluation and Prediction**: Evaluation of the models using appropriate metrics, and final model selection for predicting customer churn

You can access the machine learning notebook here:  
[Machine Learning Algorithms Notebook](https://github.com/dangwthao/Telecom-Customers-Churn-Prediction/blob/main/Fitting_model.ipynb)

### Algorithms Implemented:
- **Logistic Regression**
- **Random Forest Classifier**

### Evaluation Metrics:
- AUC ROC
- Accuracy
- Precision, Recall, and F1 Score


## How to Run

1. Clone the repository to your local machine:
    ```bash
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd Telecom-Customer-Churn-Prediction
    ```
3. Run the Jupyter notebooks in the order specified to explore and model the data.

---

## Conclusion

This project demonstrates how to approach customer churn prediction using both unsupervised (K-Means Clustering) and supervised (Logistic Regression and Random Forest) learning techniques. By analyzing customer behavior and characteristics, we can gain insights into which factors lead to churn and take action to retain valuable customers.
