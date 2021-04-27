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
