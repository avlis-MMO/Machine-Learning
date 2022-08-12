from sklearn.datasets import load_boston
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# This script was made to practise machine learning and get more used with its functions and such using the boston dataset
# Load dataset
boston = load_boston()

# Save it to a dataframe
df = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
print(df)
X,y = load_boston(return_X_y=True)
print(X)
print(boston['feature_names'])
pipe = Pipeline([("scale", StandardScaler()),("model", KNeighborsRegressor())])

mod = GridSearchCV(estimator=pipe, param_grid={'model__n_neighbors':[1,2,3,4,5]}, cv=5)
mod.fit(X,y)
pred = mod.predict(X)
plt.scatter(pred, y)
plt.show()
