
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

bos=load_boston()

print(bos.feature_names)

X=bos.data

y=bos.target

#KNeighborsRegressor
tree= KNeighborsRegressor()

tree.fit(X,y)

p=tree.predict(X)

plt.scatter(y,p)

plt.title("KNN Regressor")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.show()

print("KNeighborsRegressor RMSE:" , np.sqrt(mean_squared_error(y,p)))

#DecisionTreeRegressor
tree1=DecisionTreeRegressor()

tree1.fit(X,y)

p=tree1.predict(X)

print("DecisionTreeRegressor RMSE:",np.sqrt(mean_squared_error(y,p)))

plt.scatter(y,p)

plt.title("Decision Tree Regressor")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.show()
