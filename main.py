import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
data = pd.read_csv('/content/dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Display information about the dataset
print(data.info())

# Display statistical summary of the dataset
print(data.describe())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Create a scatter plot
plt.figure(figsize=(15, 6))  # Adjusted figure size for clarity

# Scatter plot for Price vs Mileage
plt.subplot(1, 3, 1)
plt.scatter(data['price'], data['mileage'], alpha=0.5)
plt.title('Price vs Mileage')
plt.xlabel('Price')
plt.ylabel('Mileage')

# Scatter plot for Price vs Cylinders
plt.subplot(1, 3, 2)
plt.scatter(data['price'], data['cylinders'], alpha=0.5, color='orange')
plt.title('Price vs Cylinders')
plt.xlabel('Price')
plt.ylabel('Cylinders')

# Scatter plot for Price vs Year
plt.subplot(1, 3, 3)
plt.scatter(data['price'], data['year'], alpha=0.5, color='green')
plt.title('Price vs Year')
plt.xlabel('Price')
plt.ylabel('Year')

# Check if there is a 'fuel' column in your dataset
if 'fuel' in data.columns:
    fuel_counts = data['fuel'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(fuel_counts.index, fuel_counts.values, color='purple')
    plt.xlabel('Fuel Type')
    plt.ylabel('Count')
    plt.title('Count of Cars by Fuel Type')
else:
    print("The dataset does not contain a 'fuel' column.")

plt.tight_layout()  # Adjust subplots to fit into figure area
plt.show()

# Feature Engineering
data['fuel'] = data['fuel'].map({'Petrol': 0, 'Diesel': 1, 'Hybrid': 2, 'Electric': 3})
data['doors'] = data['doors'].map({'Two': 0, 'Four': 1})

# Drop rows where 'price', 'mileage', or 'cylinders' are missing
data = data.dropna(subset=['price', 'mileage', 'cylinders'])

# Check the shape of the dataset after dropping missing values
print("Shape of dataset after dropping missing values:", data.shape)

# Feature Selection
X = data[['year', 'cylinders', 'mileage']]
y = data['price']

# Check if X and y are empty
if X.empty or y.empty:
    print("X or y is empty. Please check your data.")
else:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Linear Regression - Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Linear Regression - R-squared:", r2_score(y_test, y_pred))

    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    print("Ridge Regression - Mean Squared Error:", mean_squared_error(y_test, y_pred_ridge))
    print("Ridge Regression - R-squared:", r2_score(y_test, y_pred_ridge))

    # Lasso Regression
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    print("Lasso Regression - Mean Squared Error:", mean_squared_error(y_test, y_pred_lasso))
    print("Lasso Regression - R-squared:", r2_score(y_test, y_pred_lasso))

    # Random Forest Regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Regression - Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
    print("Random Forest Regression - R-squared:", r2_score(y_test, y_pred_rf))

    # Polynomial Regression
    degree = 2  # You can change this to any degree you want
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit the model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # Make predictions
    y_pred_poly = poly_model.predict(X_test_poly)
    print("Polynomial Regression - Mean Squared Error:", mean_squared_error(y_test, y_pred_poly))
    print("Polynomial Regression - R-squared:", r2_score(y_test, y_pred_poly))
    
