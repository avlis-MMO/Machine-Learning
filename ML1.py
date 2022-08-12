from operator import index
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import statistics

# In this script we create our own alghoritmo to indentify what kind of flower it is using one of the many datasets of sklearn
# We load the data with load_iris from sklearn
data = load_iris()

features = data['data'] # 0-sepal_len, 1-sepal_wit, 2-petal_len. 3-petal_wit has lengths
feature_names = data['feature_names'] # has the names of the variables

target = data['target'] # Has the different plants
plant_name=[]
for i in target:
    if i == 0:
        plant_name.append('Setosa')
    elif i == 1:
        plant_name.append('Versicolor')
    else:
        plant_name.append('Virginica')
columns = ['Flowers Name'] + feature_names

data_t = np.concatenate((np.array([plant_name]).T, features), axis=1)
index_values = np.arange(0,len(target))
df = pd.DataFrame(data = data_t, index=index_values, columns=columns)

for t,marker,c in zip(range(3),">ox","rgb"): #(triangles) - Iris Setosa,Iris Versicolor - (circle) and Iris Virginica - "x"
 # We plot each class on its own to get different colored markers
    plt.scatter(features[target == t,2], features[target == t,3], marker=marker, c=c)
plt.show()

# use numpy operations to get setosa features
max_setosa = df.loc[df['Flowers Name'] == 'Setosa']['petal length (cm)'].max()
min_non_setosa = df.loc[df['Flowers Name'] != 'Setosa']['petal length (cm)'].min()

print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))

#if df['petal length (cm)'] < 2: print('Setosa')

# Get datafrma with info of the other two flowers
Versicolor = df.loc[df['Flowers Name'] == 'Versicolor'].drop('Flowers Name', axis=1).astype(float)
Virginica = df.loc[df['Flowers Name'] == 'Virginica'].drop('Flowers Name', axis=1).astype(float)

mean_Versi = []
mean_Virgi = []

# Get the mean value of each of the properties of the two flowers
for i in range(Versicolor.shape[1]):
    mean_Versi.append(Versicolor.iloc[:,i].mean())
    mean_Virgi.append(Virginica.iloc[:,i].mean())

# Get the mean value of the propersties of both flowers and use it and save it in as list to use as thresold
Thres = [statistics.mean(t) for t in zip(mean_Versi, mean_Virgi)]

best_acc = 1000000

# See which thres is more accurate by couting how many values of each flower are on the wrong side of the thres, the one with less is the most accurate
for i in range(len(Thres)):
    if mean_Versi[i] >= Thres[i]:
        count1 = (Versicolor.iloc[:,i] < Thres[i]).sum()
        count2 = (Virginica.iloc[:,i] >= Thres[i]).sum()
    else:
        count1 = (Versicolor.iloc[:,i] >= Thres[i]).sum()
        count2 = (Virginica.iloc[:,i] < Thres[i]).sum()
    acc = count1 + count2

    if acc < best_acc:
        best_acc = acc
        vab = i
        
#if petal_width > Thres[vab] print(Virginica)
    


