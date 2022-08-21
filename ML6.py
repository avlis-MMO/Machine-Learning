import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import  Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from bs4 import BeautifulSoup as bs
import requests

# This python script will extraxt the data with the heights and weights of 25000 people and use an alghoritmo to try predict the weight based on the height
# For this a lot some sclares were tested and some models were also tested to find the best one
Height = []
Weight = []

# First we scrap the information from the webpage
url = 'http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html'
web_site = requests.get(url)
soup = bs(web_site.text, 'html.parser')
table = soup.find('tbody')

# Run through the table with the information
for j,line in enumerate(table.find_all('tr')):
    # Skip the first line that contains text
    if j == 0:
        continue
    for i,col in enumerate(line.find_all('td')):
        # Skip the first column that contais the index
        if i == 0:
            continue
        elif i == 1:
            Height.append(col.text)
        else:
            Weight.append(col.text)

# Store all the information on a dataframe
df = pd.DataFrame(list(zip(Height, Weight)),
               columns =['Height', 'Weight'])
df = df.astype(float)

# Create the X data and y data to train the alghoritmo
X_data = np.array(df['Height']).reshape(-1,1)
y_data = np.array(df['Weight']).reshape(-1,1)

# Use minmax scaler to convert all the values to the same scale and units, after testing this was the scaler tat provided the lowest mean squared error
scaler = MinMaxScaler()
X = scaler.fit_transform(X_data)
y = scaler.fit_transform(y_data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Both the ridge model and linear regression provided the similar results, but ridge was chosen because it has more hyperparameters to change and havent used before
model = Ridge()

# Test values for the different parameters and find the best ones based on the best mean squared error usinf grid search
params = {"alpha":[0.001,0.01,0.1,1,10],'fit_intercept':[True, False], 
'solver':['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
mod = GridSearchCV(Ridge(), param_grid = params, scoring='neg_mean_squared_error')

# Show the error and the best parameters
result = mod.fit(X_train, y_train)  
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# Draw the line showing the prediction of the alghoritmo
pred = mod.predict(X)
pred = scaler.inverse_transform(pred)
plt.scatter(X_data, y_data)
plt.plot(X_data, pred, color = 'red')
plt.show()