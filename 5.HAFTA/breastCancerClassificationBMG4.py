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
