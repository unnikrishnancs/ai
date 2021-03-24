#Implement a feed forward neural network for solving pima Indians 
#dataset from the Questions folder
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy

Outputs=numpy.array(["C1","C2"])

data=pd.read_csv("d:/freelance/handson/2_mlp/pimaindians_1.csv",header=None)
print("Top 5 rows:\n %s"%data.head())
print("\n")

#Define X (Feature Vector/Independent Variable) and y (Target/Dependent Variable)
X=data.iloc[:,:-1].values #Converts Dataframe to Numpy array (to use SLICING else TypeError when using Slicing)  
y=data.iloc[:,-1]


#======================
#======================
# PRE-PROCESSING
#======================
#======================

#Scaling
print("Before Scaling...")
print("X[0:1] :\n %s"%X[0:1])
sc=StandardScaler()
X=sc.fit_transform(X)
print("After Scaling...")
print("X[0:1] :\n %s"%X[0:1])
#print("Mean of col 0 :\n %d"% ????)

#Convert output to OHE form (One Hot Encoded). MUST for multi-class
y=to_categorical(y)

#Split data into Test and Train set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)

model=Sequential()

#Add Layers
model.add(Dense(1,input_shape=(8,),activation="relu")) #First Hidden Layer (Input Layer also specified having 8 nodes)
model.add(Dense(2,activation="softmax")) #Output layer having two nodes

#Compile model [Optimizer = ADAM]
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#Fit model
#model.fit(X_train,y_train,epochs=20,verbose=1)
history = model.fit(X_train, y_train, validation_split=0.1, epochs=20,verbose=2) #batch_size=10,

# list all data in history
print("History Keys:\n %s"%history.history.keys())

#Plot "Accuracy" history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(['Train','Validation'],loc='upper left')
plt.show()

#Plot "Loss" history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(['Train','Validation'],loc='upper left')
plt.show()

#Evaluate model
results=model.evaluate(X_test,y_test)
print("===================")
print("Test Accuracy and Loss")
print("===================")
print("Result of evaluate (overall): \n %s"%results)
print("Loss: \n %.3f"%float(results[0]))
print("Accuracy: \n %.2f"%(float(results[1])*100))

print("\n")

#Predict
print("===================")
print("Prediction for X_test: ")
print("===================")
y_pred=model.predict(X_test)
#[print(a,b) [for a in y_test] [for b in y_pred]]
[print(b,"Index corres. to Max. value:%d"%numpy.argmax(b),"Predicted Class:%s"%Outputs[numpy.argmax(b)]) for b in y_pred]
