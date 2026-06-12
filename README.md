# 📊 Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer churn occurs when customers stop using a company's products or services, leading to revenue loss and reduced business growth. Predicting churn enables businesses to identify at-risk customers and implement strategies to improve customer retention.

In this project, the **Telco Customer Churn Dataset** is analyzed using Python and Machine Learning techniques. The data is preprocessed, explored, and used to build a **Random Forest Classifier** that predicts whether a customer is likely to churn.

The project aims to answer questions such as:

* Which customer characteristics are associated with churn?
* How can missing and categorical data be prepared for machine learning?
* How accurately can customer churn be predicted?
* Which customers are most likely to leave the company?

---

# 📊 Dataset Description

The dataset contains information about telecom customers, including demographic details, subscribed services, billing information, and churn status.

## 📋 Important Columns Description

| Column Name          | Description                                               | Data Type       |
| -------------------- | --------------------------------------------------------- | --------------- |
| **customerID**       | Unique identifier for each customer                       | Text            |
| **gender**           | Gender of the customer                                    | Text            |
| **SeniorCitizen**    | Whether the customer is a senior citizen (0/1)            | Number          |
| **Partner**          | Whether the customer has a partner                        | Text            |
| **Dependents**       | Whether the customer has dependents                       | Text            |
| **tenure**           | Number of months the customer has stayed with the company | Number          |
| **PhoneService**     | Availability of phone service                             | Text            |
| **MultipleLines**    | Whether multiple phone lines are used                     | Text            |
| **InternetService**  | Type of internet service (DSL, Fiber Optic, None)         | Text            |
| **OnlineSecurity**   | Online security subscription status                       | Text            |
| **OnlineBackup**     | Online backup subscription status                         | Text            |
| **DeviceProtection** | Device protection service status                          | Text            |
| **TechSupport**      | Technical support subscription status                     | Text            |
| **StreamingTV**      | Streaming TV subscription status                          | Text            |
| **StreamingMovies**  | Streaming movie subscription status                       | Text            |
| **Contract**         | Contract type (Month-to-month, One year, Two year)        | Text            |
| **PaperlessBilling** | Whether paperless billing is enabled                      | Text            |
| **PaymentMethod**    | Customer's payment method                                 | Text            |
| **MonthlyCharges**   | Monthly amount charged to the customer                    | Number          |
| **TotalCharges**     | Total amount charged during the customer's tenure         | Number          |
| **Churn**            | Whether the customer left the company (Yes/No)            | Target Variable |

---

# 🛠️ Project Workflow

## 1️⃣ Import Required Libraries

The project uses the following Python libraries:

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## 2️⃣ Load the Dataset

The Telco Customer Churn dataset is loaded into a Pandas DataFrame for analysis and preprocessing.

## 3️⃣ Data Cleaning and Preprocessing

The following preprocessing steps are performed:

* Checked for missing and null values.
* Converted the `TotalCharges` column into numeric format.
* Replaced missing values with the median value.
* Encoded categorical variables using **Label Encoding**.
* Removed unnecessary columns such as `customerID`.
* Split the dataset into training and testing sets.
* Applied **StandardScaler** for feature scaling.

---

# 🤖 Machine Learning Model

The project uses the **Random Forest Classifier**, an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

The trained model predicts whether a customer is likely to churn based on their service usage and account information.

---

# 📈 Model Evaluation

The model performance is evaluated using:

* ✅ Accuracy Score
* ✅ Confusion Matrix
* ✅ Precision
* ✅ Recall
* ✅ Classification Performance

The achieved model accuracy is approximately **78%**, indicating that the model can effectively distinguish between customers who are likely to churn and those who are likely to stay.

---

# 🔍 Key Insights

* 📉 Customers with month-to-month contracts are more likely to churn.
* 💳 Payment methods and billing preferences have a significant impact on customer retention.
* 🌐 Internet service type influences churn behavior.
* 📆 Customers with shorter tenure are generally more likely to leave the company.
* 📊 Proper preprocessing and feature scaling improve machine learning performance.
* 🤖 The Random Forest model provides reliable predictions for identifying potential churners.

---

# 💻 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# 🎯 Conclusion

This project demonstrates how Machine Learning can be applied to predict customer churn and support business decision-making. By combining data preprocessing, exploratory analysis, and the Random Forest algorithm, the model helps identify customers at risk of leaving, enabling businesses to take proactive retention measures.

The insights generated from this analysis can assist telecom companies in improving customer satisfaction, reducing churn rates, and increasing long-term profitability.
