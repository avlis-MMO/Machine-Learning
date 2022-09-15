import numpy as np
import pandas as pd
import seaborn as sns
import os, itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# This python script will try to learn the best alghoritmo to try and predict if a fire 
# was extinguish with sound based on the size, the distance and frequency.

# Prittify cm
def prittify_CM(cm):

    cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # Lets prittify
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    classes = False
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title='Confusion Matrix', xlabel = 'Predictd Label', ylabel = 'True Label',
        xticks = np.arange(n_classes), yticks= np.arange(n_classes), xticklabels = labels,
        yticklabels = labels)

    # Set x-axis to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(20)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min())/2

    # Plot the text on each cell
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i,j]*100:.1f}%)", 
                horizontalalignment="center", color="white" if cm[i,j] > threshold else "black",
                size = 15)
    plt.show()

# 1. Get the data
directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(directory_of_python_script,'Acoustic_Extinguisher_Fire_Dataset.xlsx'))

# 2. Check data
# Status 0 - not extinguish, 1 - extinguish
print(df.head())

# Check for null values
df.info()

# Check more detailed information
print(df.describe(include='all')) 

# Check the correlation between the columns
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.show()

# Check the unique number of sizes
print(df['SIZE'].value_counts(ascending=True)) 

# 3. Clean and preprocess data
plt.bar(df['DESIBEL'].value_counts().index.to_list(), df['DESIBEL'].value_counts().values)
plt.show()                                              # There are a lot of different values and the distribution isnt unifrom
df['DESIBEL_BAND']=pd.cut(df['DESIBEL'],5)              # and has outliers, it is better to organize it by bands and then replace 
df['DESIBEL_BAND'] = pd.Categorical(df['DESIBEL_BAND']) # it to ordinals, this will make the class more efficient
df['DESIBEL'] = df['DESIBEL_BAND'].cat.codes

# We can drop the desibel band column now
df.drop('DESIBEL_BAND', axis=1, inplace=True)
df.drop('DESIBEL', axis=1, inplace=True)

# The freq is uniformaly distributed 
df['FREQUENCY'].value_counts()


plt.bar(df['AIRFLOW'].value_counts().index.to_list(), df['AIRFLOW'].value_counts().values)
plt.show()                                              # The air flow is not uniformally distributed and has a lot of 
df['AIRFLOW_BAND']=pd.cut(df['AIRFLOW'],5)              # unique values, lets bin it and then transform it to ordinals
df['AIRFLOW_BAND'] = pd.Categorical(df['AIRFLOW_BAND'])
df['AIRFLOW'] = df['AIRFLOW_BAND'].cat.codes

# We can drop the air flow band column now
df.drop('AIRFLOW_BAND', axis=1, inplace=True)

# Lets one hot encode the fuels
df = pd.get_dummies(df, columns=['FUEL'])


# 4. Start creating model
X = df.loc[:,df.columns!='STATUS'].values
y = df.pop('STATUS')

scaler = MinMaxScaler() # Because features dont have normal distribution

# Split data into train and test
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3,shuffle=True, random_state=42)

# Fit and transform the data
X_train_normal = scaler.fit_transform(X_train)
X_test_normal = scaler.transform(X_test)

# Create model
model = XGBClassifier(random_state = 42) # Cause is one of the best classifiers

model.fit(X_train_normal, y_train)
y_pred = model.predict(X_test_normal)
print(accuracy_score(y_test,y_pred) *100) # The accuracy is extremly high

# Check confusion matrix
cm = confusion_matrix(y_test, y_pred)
prittify_CM(cm)
