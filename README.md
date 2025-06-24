# Credit Risk Classification using Random Forest and XGBoost

## Project Overview

This project aims to build and compare machine learning models to classify credit applicants into low-risk ("good") and high-risk ("bad") categories based on the German Credit Data dataset from the UCI Machine Learning Repository. The goal is to demonstrate practical data science techniques on a manageable, real-world dataset relevant to the financial industry.

## Dataset

The German Credit Data dataset contains information about credit applicants, including attributes such as age, credit amount, duration, employment status, and purpose of the loan. The dataset is moderately sized and includes both numerical and categorical variables.

## Exploratory Data Analysis (EDA)

The EDA process involved:

- Analyzing the distribution of the target variable (credit risk) to understand class imbalance.
- Visualizing the distribution of key features grouped by credit risk, such as age, credit amount, and loan purpose.
- Investigating correlations among numerical variables using a heatmap.
- Examining categorical features in relation to the target variable.

## Modeling

Two classification models were implemented and evaluated:

- **Random Forest Classifier**
- **XGBoost Classifier**

Both models were trained on the dataset using a standard train-test split. Performance was assessed using metrics such as precision, recall, F1-score, accuracy, and ROC AUC.

### Results Summary

| Metric                     | Random Forest | XGBoost |
|----------------------------|----------------|----------|
| Precision (class 0 - bad)  | 0.55           | 0.49     |
| Recall (class 0 - bad)     | 0.64           | 0.66     |
| F1-score (class 0 - bad)   | 0.59           | 0.56     |
| Precision (class 1 - good) | 0.83           | 0.82     |
| Recall (class 1 - good)    | 0.77           | 0.69     |
| F1-score (class 1 - good)  | 0.80           | 0.75     |
| Accuracy                   | 0.73           | 0.68     |
| Macro Avg F1-score         | 0.70           | 0.66     |
| Weighted Avg F1-score      | 0.74           | 0.69     |
| ROC AUC                    | 0.77           | 0.76     |

## Observations

- Random Forest shows slightly better overall performance and balanced results across both classes.
- XGBoost provides better recall for the high-risk class but slightly lower overall accuracy.
- Both models achieve ROC AUC scores above 0.75, indicating a reasonable ability to distinguish between risk classes.

## Limitations and Future Work

Initial models were intentionally kept simple to establish a realistic baseline. Further improvements can be achieved via hyperparameter tuning, feature engineering, and class balancing techniques. Advanced methods such as ensemble learning or deep learning could also be explored to enhance predictive performance.

## Conclusion

This project demonstrates the application of standard classification algorithms to a real-world credit risk problem. It highlights the trade-offs between different evaluation metrics and the importance of balancing precision and recall, especially for the minority class in an imbalanced dataset. The German Credit Data provides a practical foundation for understanding credit risk assessment in financial institutions.
