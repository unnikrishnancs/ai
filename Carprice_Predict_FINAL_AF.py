######################
######################
# Import necc.packages
######################
######################

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Import CSV file
#data=pd.read_csv("D:/cars_sampled_colrearr.csv")
data=pd.read_csv("/home/ai16/project/ml_proj/cars_sampled_colRearr.csv")


####################
####################
# Data Preprocessing
####################
####################

#Drop unnecc columns....name,dateCrawled,dateCreated ,lastSeen
cols=['name','dateCrawled','dateCreated' ,'lastSeen','postalCode']
data=data.drop(cols,axis=1) 

#Remove duplicates
data.drop_duplicates(keep='first',inplace=True)

#create subset of data after filtering above...remove very low and high values
data=data[
          (data.yearOfRegistration<=2018)
         & (data.yearOfRegistration>=1950)
         & (data.price<=150000)
         & (data.price>=100)
         & (data.powerPS<=500)
         & (data.powerPS>=10)
         ]

#Determine the age of car
data['monthOfRegistration']/=12 
data['Age']=(2018-data['yearOfRegistration'])+data['monthOfRegistration']
#data['Age']=round(data['Age'],2)

#data=data.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)
data=data.drop(['yearOfRegistration','monthOfRegistration'],axis=1)

#Identify variables that are insignificant and drop
data=data.drop(['seller','offerType'],axis=1)

#Update missing values..use mode for object and median for numeric
data_filtered=data.apply(lambda x:x.fillna(x.mean()) \
                        if (x.dtype=='float') else \
                        x.fillna(x.value_counts().index[0]))

#convert categorical to numeric
data_filtered=pd.get_dummies(data_filtered,drop_first=True)

#Define X and y
X2=data_filtered.drop(['price'],axis='columns',inplace=False)

y2=data_filtered['price']


##################
#Applying scaling
##################

# Standard Scalar
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

#On X
X2=sc.fit_transform(X2)

# Generate train and test data
X_train1,X_test1,y_train1,y_test1=train_test_split(X2,y2,test_size=0.2,random_state=3)

########################
####Linear Regression###
########################

print "==========LINEAR REGRESSION======="

lgr2=LinearRegression(fit_intercept=True)

# Fit the model
model_lin2=lgr2.fit(X_train1,y_train1)

#Predict the test data
cars_prediction_lin2=lgr2.predict(X_test1)

#Calculate RMSE 
lin_rmse2=np.sqrt(mean_squared_error(y_test1,cars_prediction_lin2))
print("RMSE      -> " + str(lin_rmse2))

#Calculate R Squared value
r2_lin_test2=model_lin2.score(X_test1,y_test1)
print("R Squared -> "+str(r2_lin_test2))
print "\n"

#Regression Diagnoistics
fob=plt.figure(figsize=(6,3))
spobj=fob.add_subplot(111)
spobj.scatter(x=cars_prediction_lin2,y=y_test1)
spobj.set_title('Linear Regression')
spobj.set_xlabel('Predicted Value ->')
spobj.set_ylabel('Actual Value ->')
plt.show()


############
#Random Forest
############

print "==========RANDOM FOREST======="

#Take time (around 2 mts) if n_estimators/trees are 100
#rf2=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,
#                         min_samples_split=10,min_samples_leaf=4,random_state=1)

rf2=RandomForestRegressor(n_estimators=10,max_features='auto',max_depth=100,
                         min_samples_split=10,min_samples_leaf=4,random_state=1)

# Fit the model
model_rf2=rf2.fit(X_train1,y_train1.ravel())

#Predict the test data
cars_prediction_rf2=rf2.predict(X_test1)


#Calculate RMSE 
rf_rmse2=np.sqrt(mean_squared_error(y_test1,cars_prediction_rf2))
print("RMSE      -> " + str(rf_rmse2))

#Calculate R Squared value
r2_rf_test2=model_rf2.score(X_test1,y_test1)
print("R Squared -> " + str(r2_rf_test2))


#Regression Diagnoistics
fob=plt.figure(figsize=(6,3))
spobj=fob.add_subplot(111)
spobj.scatter(x=cars_prediction_rf2,y=y_test1)
spobj.set_title('Random Forest')
spobj.set_xlabel('Predicted Value ->')
spobj.set_ylabel('Actual Value ->')
plt.show()


