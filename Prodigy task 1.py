# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv(r"C:\Users\Kavya Bhatt\Downloads\train.csv")  # Update the file path accordingly


# Print column names
print("Column Names:", df.columns)

# Preprocessing: Select relevant features and handle missing values
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
if target in df.columns:
    df = df[features + [target]].dropna()
else:
    print(f"Column '{target}' not found in the dataset.")

# Splitting the dataset into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Coefficients of the model
print('Coefficients:', model.coef_)
