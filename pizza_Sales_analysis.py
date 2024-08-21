import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
df=pd.read_csv(r"/content/Data Model - Pizza Sales-CSV.csv")
df.shape
df.info()
df["order_time"] = df["order_time"].astype("string")
df[["hour","minute","second"]] = df["order_time"].str.split(":",expand=True)
df["hour"].value_counts().sort_index()
totalsales=df["total_price(INR)"].sum()
print(totalsales , "rupees")
df['order_day'] = pd.to_datetime(df['order_date'], errors='coerce')
df["order_day"] = df["order_day"].dt.day_name()
df["order_day"].value_counts().sort_index()
px.histogram(df,x="order_day",color='pizza_category')
px.histogram(df,x="pizza_category",color="pizza_name")
top5 = df.groupby("pizza_name")["quantity"].count().sort_values(ascending=False).head(5)
top5
spread=df.groupby(["order_day","pizza_category"])["quantity"].sum().unstack()
spread=spread.plot(marker = ".")
spread2 = df.groupby(["order_day", "pizza_category"])["quantity"].sum().unstack()
spread2 = spread2.plot.bar()
lowest=df.loc[df['pizza_category'] == "Chicken"]
lowest=lowest.groupby("pizza_name")["quantity"].sum().sort_values(ascending=True)
lowest
df.pizza_size.value_counts()
px.histogram(df,x="pizza_size",color="pizza_category")

# machine learning model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r"/content/Data Model - Pizza Sales-CSV.csv")

# Data preprocessing
df["order_day"] = pd.to_datetime(df['order_date'], errors='coerce')
df["order_day"] = df["order_day"].dt.day_name()

# Convert categorical columns to numerical
label_encoder = LabelEncoder()
df['pizza_size'] = label_encoder.fit_transform(df['pizza_size'])
df['pizza_category'] = label_encoder.fit_transform(df['pizza_category'])
df['order_day'] = label_encoder.fit_transform(df['order_day'])
df['pizza_name'] = label_encoder.fit_transform(df['pizza_name'])

# Define features and target
X = df[['pizza_size', 'pizza_category', 'order_day', 'pizza_name']]
y = df['total_price(INR)']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the metrics
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Evaluation based on MSE and R2
if mse < 5000 and r2 > 0.8:
    print("The model is performing well with a low MSE and high R² score.")
    print("Recommendation: This model is suitable for making predictions.")
elif mse < 10000 and r2 > 0.6:
    print("The model is performing decently with a moderate MSE and acceptable R² score.")
    print("Recommendation: This model is acceptable, but there is room for improvement.")
else:
    print("The model is performing poorly with a high MSE and/or low R² score.")
    print("Recommendation: Consider improving the model by trying other algorithms or tuning hyperparameters.")

# Prediction for new data
new_data = [[2, 1, 3, 5]]  # Modify this based on actual values
predicted_sales = model.predict(new_data)
print(f"Predicted Sales: {predicted_sales[0]} INR")
