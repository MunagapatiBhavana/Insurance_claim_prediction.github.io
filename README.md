# 🚀 Insurance Claim Prediction using Machine Learning

## 🧠 Problem Statement

Insurance companies face challenges in efficiently processing and predicting claim outcomes due to the complexity of customer profiles and policy details. This project aims to build a predictive model that accurately forecasts whether a claim will be approved based on customer and policy-related features.

## 👩‍🏫 About the Project

This project was developed as part of an assignment under the **Mentor Together** program, under the valuable guidance of my mentor. The goal was to gain hands-on experience with real-world machine learning workflows including data cleaning, model training, testing, and performance evaluation.

## The Assignment:
The dataset of 1000 records and the Data dictionary that explains all 76 attributes in a record have been uploaded.  

You should try any three of all four classification methods discussed in class, namely, Decision Trees, Artificial Neural Network, Naïve Bayesian Method, and Logistic regression and thus build four classification models. For each method, try to make the model as perfect as you can. 

Then present a comparison, both pros and cons, of the quality of the models developed as applied to the given data set and recommend the model which you considered the best among the three you developed. Also, set the expectation of the accuracy of your model.

You can use XLMiner for the project. 

## 🎯 Objective

Build a machine learning model using 4 different algorithms to predict insurance claim outcomes.

## 📂 Project Structure

- `data/` - contains raw and cleaned datasets
- `notebooks/` - Jupyter notebooks used for EDA and training
- `src/` - contains source code (model training, utilities)
- `models/` - saved machine learning models
- `output/` - final predictions
- `requirements.txt` - Python dependencies

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook

## 📊 Model Used

- Decision Trees
- Artificial Neural Network
- Naïve Bayesian Method
- Logistic regression
- Random Forest Classifier with tuning and evaluation using accuracy score

## ✅ Results

- Achieved an accuracy of **93%** on training data and **95%** on test data after preprocessing

## 📌 Future Improvements
Use advanced models like XGBoost

Handle class imbalance

Create an API for model deployment

## Project Structure

Insurance Claim Prediction/
├── README.md

├── Problem_statement.doc

├── data/

| ├── data_dictionary

│ ├── train_raw_dataset.csv

│ ├── train_cleaned_dataset.xlsx

│ ├── test_raw_dataset.csv

│ └── test_cleaned.xlsx

├── notebooks/

│ └── insurance_policy holder.ipynb

├── src/

│ └── model.py

│ └── utils.py

├── models/

│ └── random_forest_model.pkl

├── output/

│ └── test_predictions.csv

└── .gitignore

🙏 Acknowledgements
Special thanks to my mentor A.V. Sridhar sir from Wipro, Hyderabad and the Mentor Together program for providing continuous support, feedback, and learning opportunities throughout this project.
