import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_excel('titanic.xlsx', sheet_name='Sheet1')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Data Preprocessing
# Handling missing values
imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
test_data['Age'] = imputer.transform(test_data[['Age']])
train_data['Fare'] = imputer.fit_transform(train_data[['Fare']])
test_data['Fare'] = imputer.transform(test_data[['Fare']])

# Fill Embarked missing values with the mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation on Training Set
y_pred_train = model.predict(X_train)
print("Training accuracy:", accuracy_score(y_train, y_pred_train))

# Predict survival on test set
test_data['Survived'] = model.predict(X_test)

# Prepare submission file
submission = test_data[['PassengerId', 'Survived']]
submission.to_csv('submission.csv', index=False)
