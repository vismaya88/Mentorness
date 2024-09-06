import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Super_Store_data.csv', encoding='ISO-8859-1')

# Convert Order Date and Ship Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

# Extracting additional date features
df['Order_Year'] = df['Order Date'].dt.year
df['Order_Month'] = df['Order Date'].dt.month
df['Order_DayOfWeek'] = df['Order Date'].dt.dayofweek

# Dropping the original datetime columns
df.drop(['Order Date', 'Ship Date'], axis=1, inplace=True)

# Additional Feature Engineering
df['Discount_Quantity_Interaction'] = df['Discount'] * df['Quantity']

# Encoding categorical columns
label_encoders = {}
for column in ['Ship Mode', 'Segment', 'Country', 'Region', 'Category', 'Sub-Category', 'City', 'State']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Dropping unnecessary columns
df.drop(['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name'], axis=1, inplace=True)

df.fillna(df.median(), inplace=True)  

print(df.dtypes)  

# Splitting data into features and target
X = df.drop(['Sales'], axis=1)
y = df['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='r2', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Making predictions on the test set
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# Plotting Feature Importance
feature_importances = grid_search.best_estimator_.feature_importances_
sorted_idx = np.argsort(feature_importances)
plt.figure(figsize=(10, 8))
plt.barh(X.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance from Gradient Boosting Regressor")
plt.show()

# Residuals Plot
plt.figure(figsize=(8, 6))
sns.residplot(x=y_test, y=y_test - y_pred, lowess=True, color="g")
plt.xlabel("Actual Sales")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.show()

# Actual vs Predicted Sales Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Sales Trend Over Time
plt.figure(figsize=(10, 6))
df.groupby('Order_Year')['Sales'].sum().plot(kind='line', marker='o')
plt.title("Sales Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()
