import numpy as np
import pandas as pd
import pickle
import os
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('events.csv')

# Filter data
data_shot = data[data['event_type'] == 1]
data_shot = data_shot[data_shot['location'] != 19]

# Feature selection
X = data_shot[['time', 'side', 'bodypart', 'location', 'situation', 'assist_method', 'fast_break']]
y = data_shot['is_goal']

# Encode categorical variables
Labelx = LabelEncoder()
for col in ['side', 'bodypart', 'location', 'situation', 'assist_method', 'fast_break']:
    X[col] = Labelx.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

# Model Training
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)

# Define the estimators
estimators = [('rfc', gbc), ('lr', lr)]

# Soft Voting Classifier
soft_voting = VotingClassifier(estimators, voting='soft')
soft_voting.fit(X_train, y_train)

# Model Evaluation
print('Gradient Boosting Classifier Accuracy:', accuracy_score(y_test, gbc.predict(X_test)))
print('Logistic Regression Accuracy:', accuracy_score(y_test, lr.predict(X_test)))
print('Soft Voting Accuracy:', soft_voting.score(X_test, y_test))

# Saving the trained model
model_path = "./new_prediction_model.sav"
joblib.dump(soft_voting, model_path)