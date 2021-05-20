import pandas as pd  
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("breast-cancer-data.csv")

data.drop(['Unnamed: 32','id'],inplace = True,axis = 1)





sns.countplot(data["diagnosis"])
print(data.diagnosis.value_counts())  

data["diagnosis"] = [1 if i.strip() == "M" else 0 for i in data.diagnosis]

data.info()

describe = data.describe() 



corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()

#
threshold = 0.75
filtre = np.abs(corr_matrix["diagnosis"] > threshold)
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features with Corr Threshold 0.75")


# %% Outliers
y = data.diagnosis
x = data.drop(["diagnosis"], axis = 1)
columns = x.columns.tolist()
clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
x_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

# threshold
outlier_threshold = -2
filtre_outlier = outlier_score["score"] < outlier_threshold
outlier_index = outlier_score[filtre_outlier].index.tolist()

# %% Show on Pilots and Drop Outliers 
plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1],color="blue", s= 50, label = "Outliers")
plt.scatter(x.iloc[:,0], x.iloc[:,1],color="k", s = 3, label = "Data Points")


radius = (x_score.max() - x_score)/(x_score.max() - x_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolors = "r", facecolors = "none", label = "Outlier Scores")
plt.legend()
plt.show()


x = x.drop(outlier_index)
y = y.drop(outlier_index).values 

#%% Train Test Split   &   Standardization

test_size = 0.3

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)




scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
X_train_df["diagnosis"] = Y_train 


data_melted = pd.melt(X_train_df, id_vars = "diagnosis",
                      var_name="features",
                      value_name="value")
plt.figure()
sns.boxplot(x = "features", y = "value", hue = "diagnosis", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

#%% Basic KNN Method


knn = KNeighborsClassifier(n_neighbors=4, weights= 'distance', p=2)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
print("Basic KNN Accuracy: ", acc)
print("Confusion Matrix: ", cm)
print() 

#%% Best Parameters

def KNN_Best_Params(x_train, x_test, y_train, y_test):
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    knn_best = KNeighborsClassifier()
    grid = GridSearchCV(knn_best, param_grid, cv = 10, scoring = "accuracy")
    grid.fit(x_train, y_train)
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn_best_params = KNeighborsClassifier(**grid.best_params_)
    knn_best_params.fit(x_train, y_train)
    y_pred_test = knn_best_params.predict(x_test) 
    y_pred_train = knn_best_params.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train) 
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print("Test score: {}, Train score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: {}",cm_test)
    print("CM Train: {}",cm_train) 
    
    return grid

grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)