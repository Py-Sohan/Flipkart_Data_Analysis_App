import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the data
df1 = pd.read_csv("flipkart_sales.csv")
df = df1.groupby(["Product Name", "Category"])["Total Sales (INR)"].sum().reset_index()

# Prepare the data for training
x = df.drop(columns=["Total Sales (INR)"])
y = df["Total Sales (INR)"]

label_cols = x.select_dtypes('object')
lb_encoders = {}
for col in label_cols:
    lb_encoders[col] = LabelEncoder()
    x[col] = lb_encoders[col].fit_transform(x[col])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=168)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title("Sales Prediction App")

col1, col2 = st.columns(2)

with col1:
    product_name = st.selectbox("Select Product Name", df["Product Name"].unique())

with col2:
    category = st.selectbox("Select Category", df["Category"].unique())

product_name_encoded = lb_encoders["Product Name"].transform([product_name])[0]
category_encoded = lb_encoders["Category"].transform([category])[0]

input_data = pd.DataFrame({
    "Product Name": [product_name_encoded],
    "Category": [category_encoded]
})

predicted_sales = model.predict(input_data)[0]

st.title(f"Predicted Total Sales: {predicted_sales:.2f} INR")

# st.write(f"Mean Absolute Error: {mae:.2f}")
# st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R^2 Score: {r2 * 100:.2f}%")