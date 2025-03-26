# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load training and testing datasets
train_data = pd.read_csv("assignment2train.csv")
test_data = pd.read_csv("assignment2test.csv")

# Define target and feature columns
target_column = 'meal'
columns_to_drop = ['meal', 'id', 'DateTime']

# Prepare training data
X_train = train_data.drop(columns=columns_to_drop, axis=1)
y_train = train_data[target_column]

# Initialize and train the model
model = DecisionTreeClassifier(max_depth=100, min_samples_leaf=10)
model.fit(X_train, y_train)

# Prepare testing data
X_test = test_data.drop(columns=columns_to_drop, axis=1)

# Make predictions
predictions = model.predict(X_test).astype(float)

# Output predictions (if needed for debugging or saving)
print(predictions)