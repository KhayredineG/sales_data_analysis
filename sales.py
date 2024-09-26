import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme(style="darkgrid")
df = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')

# Inspect data
print(df.head())
print(df.info())

# Data Cleaning
# Handle missing values, fill or drop
df.fillna(0, inplace=True)

# Exploratory Data Analysis (EDA)
# Example: Revenue trend over time
df['YEAR_ID'] = pd.to_datetime(df['YEAR_ID'])
df['Revenue'] = df['PRICEEACH'] * df['QUANTITYORDERED']

# Group by Date to see trends
revenue_trend = df.groupby('YEAR_ID')['Revenue'].sum().reset_index()

# Plot revenue trend
plt.figure(figsize=(10,6))
plt.plot(revenue_trend['YEAR_ID'], revenue_trend['Revenue'])
plt.title('Revenue Trend Over The Years', fontsize=16, weight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()

# Analyze Revenue by Product
product_revenue = df.groupby('PRODUCTLINE')['Revenue'].sum().reset_index()

# Plot top products by revenue
top_products = product_revenue.sort_values(by='Revenue', ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x='Revenue', y='PRODUCTLINE', data=top_products)
plt.title('Top 10 Products by Revenue' ,fontsize=16 , weight='bold')
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()

# Revenue Optimization (Example using Linear Regression)
# Features could include 'Price', 'Quantity', etc.
X = df[['PRICEEACH', 'QUANTITYORDERED']]
y = df['Revenue']

# Fit a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Check coefficients for optimization insights
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Use model to predict revenue (optional)
df['Predicted_Revenue'] = model.predict(X)

# Visualize actual vs predicted revenue
plt.figure(figsize=(10,6))
plt.scatter(df['Revenue'], df['Predicted_Revenue'], alpha=0.5)
plt.title('Actual vs Predicted Revenue', fontsize=16 , weight='bold')
plt.xlabel('Actual Revenue', fontsize=12)
plt.ylabel('Predicted Revenue', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()

# You can further extend this by analyzing seasonal trends, customer segmentation, etc.
