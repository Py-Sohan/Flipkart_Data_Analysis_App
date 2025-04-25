# Flipkart Sales Prediction App

This project is a Streamlit application that predicts total sales (in INR) for Flipkart products based on their name and category. It utilizes a linear regression model trained on historical sales data. The project also includes an exploratory data analysis (EDA) component in a Jupyter Notebook, providing insights into sales trends and patterns.

## Overview

![image](https://github.com/user-attachments/assets/d3f8e3dc-d387-4cfe-8eb7-4179de229f90)


The core of this project is a sales prediction tool built with Streamlit. Users can select a product name and category from dropdown menus, and the app will predict the total sales for that combination. The underlying prediction is made using a linear regression model.

Complementing the app is a Jupyter Notebook (`flipkart.pdf`) which performs EDA on the sales data. This analysis explores various aspects of the data, including:

* Top-selling products and categories[cite: 114, 115].
* Customer ratings by category[cite: 116, 117].
* Payment method distribution[cite: 119, 120].
* Sales trends and profit margins[cite: 123, 125, 126].

## Features

* **Sales Prediction:**
    * User-friendly Streamlit interface for selecting product name and category.
    * Linear regression model for predicting total sales (INR).
    * Clear display of the predicted sales value.
    * R² score to indicate model performance.
* **Exploratory Data Analysis (EDA):** (See `flipkart.pdf`)
    * Visualizations of sales data, including pie charts, bar plots, line plots, and heatmaps[cite: 111, 112, 113].
    * Analysis of product performance, customer behavior, and sales patterns.
    * Insights into average order value and its variation across payment methods[cite: 127].

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn (sklearn)
    * `LinearRegression`
    * `LabelEncoder`
    * `train_test_split`
    * `mean_absolute_error`, `mean_squared_error`, `r2_score`
* Matplotlib
* Seaborn

## Setup and Installation

1.  **Ensure you have Python installed.**

2.  **Install the required libraries:**

    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Place the `flipkart_sales.csv` file in the same directory as `app.py`.**

4.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

## Data

The application uses the `flipkart_sales.csv` dataset, which contains information about sales transactions. Key attributes include:

* Product Name [cite: 109]
* Category [cite: 109]
* Total Sales (INR) [cite: 110]

The Jupyter Notebook (`flipkart.pdf`) provides a more detailed exploration of the dataset's features.

## How to Use the App

1.  Run the Streamlit application.
2.  Select a "Product Name" and "Category" from the dropdown menus.
3.  The app will display the predicted "Total Sales (INR)" for the selected product and category.
4.  The app also shows the R² score, indicating the model's performance.

## Model Performance (from Notebook)

The linear regression model's performance is indicated by the R² score. The notebook explores different random states for the train-test split to find the best R² score, noting that "Best R^2 Score: 0.19440295459919277 with random_state: 168"[cite: 142]. Further model evaluation metrics (MAE, MSE) are also present in the notebook[cite: 143].

## Key Insights from EDA


* Electronics and Home & Kitchen categories contribute significantly to revenue[cite: 115].
* Customer ratings vary across categories, with Home & Kitchen generally having higher ratings[cite: 117].
* UPI, Debit Card, and Wallet are among the commonly used payment methods[cite: 120].
* "Educational Book" and "Laptop" are among the top-selling products[cite: 123].

## Limitations

* The linear regression model may not capture complex non-linear relationships between product characteristics and sales.
* The model's predictive power is limited by the available features (product name and category).
* The R² score suggests that the model explains a limited portion of the variance in sales.

## Further Work

* Explore more advanced machine learning models (e.g., Random Forest, Gradient Boosting) to potentially improve prediction accuracy.
* Incorporate additional features (e.g., price, customer ratings) into the model.
* Implement feature engineering techniques to create new predictive variables.
* Enhance the Streamlit app with visualizations of sales trends and predictions.
