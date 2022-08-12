from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Simple machine learning to identify the iris and to practise how to use machine learning classifiers using again the iris dataset and find the best
data = load_iris()

# Separate data from the labels
X = data['data']
labels = data['target']
y = labels

# Create model and train model
pipe = Pipeline([("scale", StandardScaler()),("model", KNeighborsClassifier())])
mod = GridSearchCV(estimator=pipe, param_grid={'model__n_neighbors':[1,2,3,4,5,6,7,8,9,10]}, cv=15)
mod.fit(X,y)
print(pd.DataFrame(mod.cv_results_))

# Predict
y_pred = mod.predict(X)

# Test to see if is overfitting
print(accuracy_score(y, y_pred)*100)
print(mean_absolute_error(y, y_pred)*100)