import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# This script will use a parkisons dataset found to create an alghoritmo to predict if it or not, and we will try to use different preprocessing functions to find the best
directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
file_name = 'parkinsons.data'
df = pd.read_csv(os.path.join(directory_of_python_script, file_name))

# Get Features and labels
features = df.loc[:,df.columns!='status'].values[:,1:]
print(features)
labels = df.pop('status')

print(df.head())
print(labels.value_counts())

scaler=StandardScaler()
x=scaler.fit_transform(features)
y=labels

#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

# Train the model
model=XGBClassifier()
model.fit(x_train,y_train)

# Get the accuracy to see which preprocessing is better
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)