import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from xgboost import XGBRegressor
import pickle

dataset = pd.read_csv("C:/Users/urmio/OneDrive/Documents/boston-housing-project/BostonHousing.csv")
#print(dataset)

#print(dataset.head())

#print(dataset.tail())

#print(dataset.shape)

#print(dataset.isnull().sum())

correlation = dataset.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
#plt.show()

X=dataset.drop('price',axis=1)
Y=dataset['price']
#print(X)

#print(Y)

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=31)
#print(X.shape,X_train.shape,X_test.shape)

model = LinearRegression()

model.fit(X_train,Y_train)

model_prediction=model.predict(X_train)

#print(model_prediction)

score1= metrics.r2_score(model_prediction,Y_train)
#print("R2 score = ",score1)

score2 = metrics.mean_absolute_error(model_prediction,Y_train)
#print("Mean absolute error score =",score2)

model2= XGBRegressor()
model2.fit(X_train,Y_train)

model_prediction=model2.predict(X_train)

score1=metrics.r2_score(model_prediction,Y_train)
#print("R2 score= ",score1)

score2= metrics.mean_absolute_error(model_prediction,Y_train)
#print("Mean absolute error score =",score2)


input=np.array([[0.04527,	0.0,	11.93,	0,	0.573,	6.120,	76.7,	2.2875,	1,	273,	21.0,	396.90, 9.08]])
print(model2.predict(input)) #actual value = 20.6

with open("model2.pkl", "wb") as file:
    pickle.dump(model2, file)