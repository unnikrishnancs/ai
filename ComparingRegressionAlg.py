"""
9)generate the box plot showing the comparison of cross validated RMSE values for the 
boston dataset. Apply any 5 regressors including Ridge and Lasso
"""

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error #,r_squared_error
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

boston=load_boston()

X=boston.data
y=boston.target

print X[0],y[0]

models=[]

models.append(("Ridge",Ridge()))
models.append(("Lasso",Lasso()))
models.append(("KNN",KNeighborsRegressor()))
models.append(("DT",DecisionTreeRegressor()))
models.append(("RF",RandomForestRegressor()))

results=[]
names=[]

#print sorted(sklearn.metrics.SCORERS.keys())

for nm,mdl in models:
 kfld=KFold(n_splits=10,random_state=7)
 v=cross_val_score(mdl,X,y,scoring='neg_mean_squared_error',cv=kfld)
 results.append(v)
 names.append(nm)
 print "Name:%s, Score: %s"%(nm,v)
 

fig=plt.figure()
fig.suptitle('Algorithm Comparison - Regression')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


