import pandas as pd
import numpy as np

#Pre-processing
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

#Algorithm
from sklearn.linear_model import LogisticRegression

data=pd.read_csv("D:/Freelance/Artifacts/banking_trnc.csv")

print("Print top 5 rows...\n")
print(data.head())
print("\n")

#----------------
#Basic validation
#----------------
print("1.Check number of rows and cols imported...\n")
print(data.shape)
print("\n")

print("2.Check imported types,non-null count,no.of cols against each type,memory usage..\n")
print(data.info())
print("\n")

#--------------------------
#Drop unnecc rows, cols
#--------------------------
#Remove duplicates
data.drop_duplicates(keep='first',inplace=True)

#Filter out inconsistent rows (outliers, very low or high values)
data=data[
          (data.duration!=0) & (data.duration<30000)
		 ]
print("New Shape after filtering..." + str(data.shape))

#Define X and y....
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#---------------
#PRE-PROCESSING
#---------------

col=[1,2,3,4,5,6,7,8,9,14]
lb=LabelEncoder()
for c in col: 
	X[:,c]=lb.fit_transform(X[:,c])
		
print("After Label Encoding....\n")
print(X[:5,:])

#Scale the inputs
sc=StandardScaler(with_mean=False)
X=sc.fit_transform(X)

#Split data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Initialize and train the model
lr=LogisticRegression()
lr.fit(X_train,y_train)

#Predict
y_pred=lr.predict(X_test)
print("\n")
print("\n")
print("Print Metrics.....")
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

