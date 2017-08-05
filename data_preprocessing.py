# importing libraries
import numpy as np  # 
import matplotlib.pyplot as plt  # create charts
import pandas as pd  # import and manage datasets

# importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# taking care of missing data
from sklearn.preprocessing import Imputer # import Imputer class, sklearn library used to make ML models
imputer = Imputer(missing_values = 'NaN', strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:,1:3]) # choose all lines of specific columns
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorial
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # label categorial data
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#For dummy variables, dont have to scale 
#For classification, dont have to scale y 