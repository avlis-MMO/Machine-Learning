import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# This python script will try to find and run the best alghoritmo to try and predict if a person survided or not the titanic based on
# the class they were, their gender and age. Different clssifers were evaluated, and scaler used was the Standard scaler it was the one
# with the best results 

# Get the titanic data file and save it as a dataframe
directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
file_name ='titanic.csv'
df = pd.read_csv(os.path.join(directory_of_python_script,file_name))

# Convert the genders to numbers so our alghoritmo can use it
df['Sex']=np.where(df['Sex']== 'male', 0, 1)

# Check if data is complete
print(df.isnull().sum())

# There is an outlier but not going to take it since it is relevant for our problem
sns.boxplot(df['Age']) 

# Get the labels and features to use to train our ml the labels are survided and not survived
labels = df['Survived']

# What can affect their survival is the class they were, the gender and age the other parameters dont matter
del df['Name'], df['Siblings/Spouses Aboard'], df['Parents/Children Aboard'], df['Fare']
features = np.array(df.loc[:,df.columns!='Survived'])

best_score = 0
# Start creating our model
# First find out which classifier is best for ou dataset
scaler=StandardScaler()

x = scaler.fit_transform(features)
y = labels

# Slipt the data
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.4, random_state=42)

# Create an array with the most popular classifiing models
models = [LogisticRegression(),KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(), XGBClassifier(), SVC()]
for model in models:

    # Train and test the model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # Use accuracy score to find best model and save it
    score = accuracy_score(y_test,y_pred) *100
    if score > best_score:
        best_model = model
        best_score = score

# The best model was KNeighbors
print("The best model is " + str(best_model)[:-2] + ': %s '%best_score)

# After finding what was the best model, decided to find the best hyperparameters for it using grid search
params = {'n_neighbors':list(range(1,10)), 'leaf_size':list(range(1,50)),'p':[1,2],'algorithm':['ball_tree', 'kd_tree', 'brute']}
mod = GridSearchCV(best_model, param_grid = params, scoring='accuracy')

# Train the model
result = mod.fit(x_train, y_train)

# Print th best score obtained and the corresponding values
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)