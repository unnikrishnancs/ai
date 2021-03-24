#Import Libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
import numpy as np
from keras.preprocessing import image

#=============
#=============
#Preprocessing
#=============
#=============

#Preprocess Training set
train_datagen=ImageDataGenerator(rescale=1./255,
							 shear_range=0.2,
							 zoom_range=0.2,
							 horizontal_flip=True)

train_set=train_datagen.flow_from_directory("CatsnDogs/training_set",
											target_size=(64,64),
											batch_size=2,
											class_mode='binary')

print("TRAIN Set class indices : %s \n"%train_set.class_indices)											
											
#Preprocess Test set (NO transformation. ONLY scaling)
test_datagen=ImageDataGenerator(rescale=1./255)

test_set=test_datagen.flow_from_directory("CatsnDogs/training_set",
											target_size=(64,64),
											batch_size=1,
											class_mode='binary')

print("TEST Set class indices : %s \n"%test_set.class_indices)	
											
#=============
#=============
#Build the CNN
#=============							
#=============

#Initialize CNN
cnn=Sequential()

#Convolution
#Tensorflow...channels_last
#cnn.add(Conv2D(filters=1,kernel_size=3,activation="relu",input_shape=[64,64,3])) 
#Theano...channels_first
cnn.add(Conv2D(filters=1,kernel_size=3,activation="relu",input_shape=[3,64,64])) 

#Pooling
cnn.add(MaxPool2D(pool_size=2,strides=2))

#Add a second Convolution layer
cnn.add(Conv2D(filters=1,kernel_size=3,activation="relu"))
cnn.add(MaxPool2D(pool_size=2,strides=2))

#Add Flattening
#(Convert output of pooled layer to 1-D vector inorder to pass it to Fully Connected Layer)
cnn.add(Flatten())

#Add Fully Connected Layers
cnn.add(Dense(units=1,activation="relu")) #units=128

#Add Output Layer
cnn.add(Dense(units=1,activation="sigmoid")) 

print("===========Neural Network Structure===========")
print(cnn.summary())

#=============
#=============
#Training the CNN
#=============							
#=============

#Compile CNN
cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#TRAIN the CNN on Training set and EVALUATE on Test set
#history=cnn.fit(x=train_set,validation_data=test_set,epochs=1)
history=cnn.fit_generator(train_set,validation_data=test_set,epochs=1)

print("history.history.keys : %s"%history.history.keys())

#=============
#=============
#Prediction
#=============							
#=============

test_image=image.load_img("CatsnDogs/single_prediction/cat_or_dog_1.jpg",target_size=(64,64))

#convert the above PIL formatted output to numpy array
test_image=image.img_to_array(test_image)

#Add addnl. dimension as first dimension
test_image=np.expand_dims(test_image,axis=0)

#Predict
result=cnn.predict(test_image)

#train_set.class_indices

if result[0][0]==1:
	prediction="Dog"
else:
	prediction="Cat"	

print("Image is of a %s"%prediction)


#Save model
cnn.save('model_catsdogs.h5')

