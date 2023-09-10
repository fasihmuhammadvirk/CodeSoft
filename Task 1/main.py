import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load your dataset from the CSV file
# Replace 'tested.csv' with the actual file path if it's not in the same directory
df = pd.read_csv('tested.csv')

# Drop columns that are unlikely to be useful for predictions
# Modify this line to drop the columns you don't need
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Fill missing values in the 'Age' column with the median age
# Modify this line if your dataset has a different column name for age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encode categorical variables (e.g., 'Sex' and 'Embarked') using one-hot encoding
# Modify this line to include your categorical columns
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Split the dataset into features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an imputer to fill missing values with the median
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training data and transform both training and test data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
print(classification_report(y_test, y_pred))

# Create a confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# Feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance")
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load your dataset from the CSV file
# Replace 'tested.csv' with the actual file path if it's not in the same directory
df = pd.read_csv('tested.csv')

# Drop columns that are unlikely to be useful for predictions
# Modify this line to drop the columns you don't need
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Fill missing values in the 'Age' column with the median age
# Modify this line if your dataset has a different column name for age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encode categorical variables (e.g., 'Sex' and 'Embarked') using one-hot encoding
# Modify this line to include your categorical columns
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Split the dataset into features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an imputer to fill missing values with the median
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training data and transform both training and test data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
print(classification_report(y_test, y_pred))

# Create a confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# Feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance")
plt.show()
