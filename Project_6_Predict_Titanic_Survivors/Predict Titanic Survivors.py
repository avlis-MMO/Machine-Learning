import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, itertools
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from xgboost import XGBClassifier

# This python script will try to find and run the best alghoritmo to try and predict if a person survided or not the titanic based on
# the class they were, their gender and age. Different clssifers were evaluated, and scaler used was the Standard scaler it was the one
# with the best results 

# Pritify confusion matrix
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

# 1. Get the titanic data file and save it as a dataframe
directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
file_name ='titanic.csv'
df = pd.read_csv(os.path.join(directory_of_python_script,file_name))

# 2. Check the data provided
print(df.head())
print('\n')
print(df.describe())

# Check if data is complete
df.isnull().sum()

# Check how many unique values age has and its distribution
print("Age unique values: ",df['Age'].nunique(), 'Youngest: ',df['Age'].min(), 'Oldest: ',df['Age'].max())
plt.bar(df['Age'].value_counts().index, df['Age'].value_counts().values)
plt.xlabel('Age')
plt.ylabel('Values per age')
plt.title("Number of ages")
plt.show() # There is value of 0.42 in age, prob there are more, and since the there are a lot of unqiue values
           # and outliers, lets store the age in bins and then give it ordinals, to make class more effic and better.
           # There seems to be outliers but lets consider them since it matters for us

# 3. Clean data
# Delete the column name and fare cause this doesnt influence if they survived or not
# The other columns are personal atributes that influence
del df['Name'], df['Fare']

# Keep the Siblings/spouse and parents/chilfren because this may have influenced their actions during the indicent
print(df[["Siblings/Spouses Aboard", "Survived"]].groupby(['Siblings/Spouses Aboard'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(df[["Parents/Children Aboard", "Survived"]].groupby(['Parents/Children Aboard'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Lets store age as bins as said before since min is ~0 and max 80
df['Age'] = pd.cut(df['Age'], 8)

print(df.head())

# 4. Preprocessing steps
# Get the labels and features to use to train our ml the labels are survided and not survived
y = df['Survived']
X = df.loc[:,df.columns!='Survived']

# Create pipeline and transformer
categorical_transformer = Pipeline([
                        ('ordinal', OrdinalEncoder()),
                        ('minmax', MinMaxScaler())
]) # Transform the age to numbers and then minmaxscale

ct = make_column_transformer(
    (MinMaxScaler(),['Pclass', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']),
    (categorical_transformer, ['Age']),
    (OneHotEncoder(handle_unknown='ignore'), ['Sex']), # Transform the sex to 0 or 1
)

# Slipt the data
x_train,x_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=42)

# Fit transform data
x_train_norm = ct.fit_transform(x_train)
x_test_norm = ct.transform(x_test)

# 5. Start creating our model
# Create an array with the most popular classifiing models
best_score = 0
models = [('LG', LogisticRegression(random_state = 42)),('KN',KNeighborsClassifier()),('DT',DecisionTreeClassifier(random_state = 42)),
          ('RF',RandomForestClassifier(random_state = 42)), ('SVC', SVC(random_state = 42)), ('XGB', XGBClassifier(random_state = 42))]
for name, model in models:

    # Train and test the model
    model.fit(x_train_norm, y_train)
    y_pred = model.predict(x_test_norm)
    # Use accuracy score to find best model and save it
    score = accuracy_score(y_test,y_pred) *100
  
    if score > best_score:
        best_model = model
        best_score = score
        name_best = name

# The best model was...
print("The best model is " + name_best + ': %s '%best_score)

# After finding what was the best model, decided to find the best hyperparameters for it using grid search
params = {'max_depth':list(range(1,9)), 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3], 'n_estimators':list(range(100,500,100)),'random_state':[42]}
mod = GridSearchCV(best_model, param_grid = params, scoring='accuracy')

# Train the model
result = mod.fit(x_train_norm, y_train)
y_pred = mod.predict(x_test_norm)
# Print th best score obtained and the corresponding values
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# Check confusion matrix
cm = confusion_matrix(y_test, y_pred) # A lot of false negatives but for ou data
prittify_CM(cm)                       # it doesnt matter