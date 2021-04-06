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
