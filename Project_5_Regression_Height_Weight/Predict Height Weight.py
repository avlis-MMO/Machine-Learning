import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import  Ridge, LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from bs4 import BeautifulSoup as bs
import requests
from sklearn.metrics import mean_absolute_error
from scipy.stats import shapiro

# This python script will extraxt the data with the heights and weights of 25000 people and use 
# an alghoritmo to try predict the weight based on the height

# Change units of height
def change_inch_cm(col):
    if pd.isnull(col) == True:
        return None
    cm = col * 2.54
    return round(cm,1)

# Change units of weight
def change_pd_kg(col):
    if pd.isnull(col) == True:
        return None
    kg = col * 0.4536
    return round(kg,1)

Height = []
Weight = []

# 1. First we scrap the information from the webpage
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

# 2. Store all the information on a dataframe
df = pd.DataFrame(list(zip(Height, Weight)),
               columns =['Height', 'Weight'])
df = df.astype(float)
plt.scatter(df['Height'], df['Weight'])
plt.show()
print(df.head())

# 3. Clean data, change its units
df['Height'] = df['Height'].apply(change_inch_cm)
df['Weight'] = df['Weight'].apply(change_pd_kg)

# 4. Visualize data for missing values and other information
df.info() # There are no null values and all are float

print(df.describe()) # There seems to be a difference between the min and max and the percentiles,
                     # there may be outliers

# 5. Since it is a regression exercise lets see the correlation between the two variables
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.show() # The correlation is only 0.5, this may because of the presence of outliers,
           # so lets investigate our data and try improve it

# 6.Lets see how is the distribution of the two features
fig, axs = plt.subplots(1,2)
sns.histplot(ax = axs[0], data = df['Height'], bins = 20)
axs[0].set_title('Hegiht')
sns.histplot(ax = axs[1], data = df['Weight'], bins = 20, color='green')
axs[1].set_title('Wegiht')
plt.show() # We can see that both height and weight are normally distributed

# Statistical test to confirm normal distribution
print(shapiro(df['Height']), shapiro(df['Weight'])) # p-value > 0.05 normal distribution

# 7. Lets check for outliers
fig, axs = plt.subplots(1,2)
sns.boxplot(ax = axs[0], data = df['Height'])
axs[0].set_title('Height')
sns.boxplot(ax = axs[1], data = df['Weight'], color='green')
axs[1].set_title('Weight')
plt.show() # We can see there are some outliers

# 8. Lets clean the outliers
# Height
# Get upper and lower bonds
data_mean, data_std = np.mean(df['Height']), np.std(df['Height'])
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off

# Identify outliers
df.loc[(df['Height'] < lower) | (df['Height'] > upper)]
# Removing outliers
df = df.loc[(df['Height'] > lower) & (df['Height'] < upper)]

# Weight
# Get upper and lower bonds
data_mean, data_std = np.mean(df['Weight']), np.std(df['Weight'])
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off

# Identify outliers
df.loc[(df['Weight'] < lower) | (df['Weight'] > upper)]
# Removing outliers
df = df.loc[(df['Weight'] > lower) & (df['Weight'] < upper)]

# 9. Check data again and correlation
print(df.describe()) # The percentiles are much closer to min and max

sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.show() # But the correlation hasnt changed much so we cant expect good results from the 
           # predictive model, but we will try either way

# 10. Start creating the model
# Create the X data and y data to train the alghoritmo
X_data = np.array(df['Height']).reshape(-1,1)
y_data = np.array(df['Weight']).reshape(-1,1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)

# Since data is in the same scale we dont need to scale it or normalizite

# Liner regression would do the job but wanted to test the ridge model
model = Ridge()

# Test values for the different parameters and find the best ones based on the best mean squared error usinf grid search
params = {"alpha":[0.001,0.01,0.1,1,10],'fit_intercept':[True, False], 
'solver':['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
mod = GridSearchCV(Ridge(), param_grid = params, scoring='neg_mean_squared_error')

# Show the error and the best parameters
result = mod.fit(X_train, y_train) 
pred = mod.predict(X_test)

print('Best Hyperparameters: %s' % result.best_params_)
print(mean_absolute_error(y_test, pred))

# Draw the line showing the prediction
plt.scatter(X_data, y_data)
plt.plot(X_data, mod.predict(X_data), color = 'red')
plt.show() # As expected the predition is not the best as seen by the line a lot of values dont fall 
