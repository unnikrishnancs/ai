#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn import preprocessing

df = pd.read_csv('/home/ai16/ml_assign/Data/bike_sharing.csv')

X = df.drop(['datetime', 'count'], axis=1)
y = df['count']

#print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

lr = LinearRegression()

lr.fit(X_train, y_train)

p = lr.predict(X_test)

print "Mean Squared Error..."
print(mean_squared_error(y_test, p))

plt.scatter(y_test, p)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

#====================
#After Scaling.....
#====================

X_scale=preprocessing.scale(X)

X_scale_train, X_scale_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2,random_state=1)

lr = LinearRegression()

lr.fit(X_scale_train, y_train)

p_afterscale = lr.predict(X_scale_test)

print "Mean Squared Error...After Scaling"
print(mean_squared_error(y_test, p_afterscale))

plt.scatter(y_test, p_afterscale)
plt.xlabel('Actual')
plt.ylabel('Predicted_AfterScale')
plt.show()


#====================
#After Normalising.....
#====================

X_norm=preprocessing.normalize(X)

X_norm_train, X_norm_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2,random_state=1)

lr = LinearRegression()

lr.fit(X_norm_train, y_train)

p_afterNorm = lr.predict(X_norm_test)

print "Mean Squared Error...After Normalizing"
print(mean_squared_error(y_test, p_afterNorm))

plt.scatter(y_test, p_afterNorm)
plt.xlabel('Actual')
plt.ylabel('Predicted_AfterNorm')
plt.show()

#====================
#After MinMax scaling.....
#====================

min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)

X_minmax_train, X_minmax_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.2,random_state=1)

lr = LinearRegression()

lr.fit(X_minmax_train, y_train)

p_afterMinMax = lr.predict(X_minmax_test)

print "Mean Squared Error...After MinMax"
print(mean_squared_error(y_test, p_afterMinMax))

plt.scatter(y_test, p_afterMinMax)
plt.xlabel('Actual')
plt.ylabel('Predicted_AfterMinMax')
plt.show()



