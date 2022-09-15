import pandas as pd
import  numpy as np
import os, itertools
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# This script will use a parkisons dataset found to create an alghoritmo to predict if 
# it or not, and we will try to use different preprocessing functions to find the best
def handle_outliers(col):
    if pd.isnull(col) == True:
        return None
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    return lower, upper

def boxplot():
    col = [col for col in df.columns]
    w=0
    fig, axs = plt.subplots(4,6)
    for i in range(4):
        for j in range(6):
            if w < 23:
                if col[w] == 'status':
                    w = w+1
                axs[i,j].boxplot(x = df[col[w]])
                axs[i,j].set_title(str(col[w]), fontsize = 10)
                axs[i,j].set_yticklabels([])
                axs[i,j].set_xticklabels([])
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
                w = w +1
            else:
                axs[i,j].remove()
    plt.show()

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
file_name = 'parkinsons.data'
df = pd.read_csv(os.path.join(directory_of_python_script, file_name))

# 2. Visualise if the data is complete and other info
print(df.head())
print(df.describe())
df.info() # All the data is numerical besides the name

# 3. Start seeing data for outliers and distribution
# Drop name cause it inst useful
df.drop('name', axis=1, inplace=True)

# Check for outliers
print(df.skew()) # Seems to be a lot of outliers
boxplot()# As seen by the plot it confirms the results from skew there are a lot of outliers

# Lets select the columns to get reed of the outliers
outliers = []
for i in range(df.shape[1]):
    if df.skew().index[i] == 'status':
        continue
    if df.skew()[i] > 0.5 or df.skew()[i] < -0.5:
        outliers.append(df.skew().index[i])

# Remove outliers
if len(outliers) > 1:
    for col in outliers:
        lower, upper = handle_outliers(col)
        df = df[(df[col] > lower) & (df[col] < upper)]

print(df.skew()) # we can see the skew is much better
boxplot()

# Check distribution
df.hist()
plt.show() #It inst normally distributed

# See the correlation to check which one affects more the status
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.show()

# 4. Start creating model
scaler=MinMaxScaler() # Set values between -1 and 1 to be in same scale
features = df.loc[:,df.columns!='status'].values
labels = df.pop('status')

# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=7)

# Fit and transform the data
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the model
model=XGBClassifier()
model.fit(x_train,y_train)

# Get the accuracy 
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100) # The accuracy is good

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred) # It got 0 false negatives wich is good since is best to test positive and be negtaive
prittify_CM(cm)                       # than the other way around