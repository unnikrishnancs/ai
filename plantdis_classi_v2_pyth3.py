#Import neccessary packages
from os import listdir
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


#Initialise few vars
EPOCHS = 10
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = '/home/ai16/project/main_project/Data/Plantvillage'
width=256
height=256
depth=3

#Function to convert images to array
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        #print(f"Error : {e}")
        print("convert_image_to_array() : Error when converting Image to Array:"+e)
        return None

#Fetch images from directory and convert each to an array...assign labels to it (folder name)
image_list, label_list = [], []
try:
    print("[INFO] Loading images from folders...")
    root_dir = listdir(directory_root)
    for plant_folder in root_dir :
        print("Processing images from folder... ->: "+ plant_folder)
        #below will return list of image files within the folder
        Images_In_Folder = listdir(directory_root+"/"+plant_folder)
        #print "------------------"
        #print("\n")

        #for image in plant_disease_folder_list[:200]:
        for image in Images_In_Folder:
                image_filename = directory_root+"/"+plant_folder+"/"+image 
                if image_filename.endswith(".jpg") == True or image_filename.endswith(".JPG") == True:
                    #print "Image filename (Abs. path) ->" + image_filename 
                    image_list.append(convert_image_to_array(image_filename))
                    label_list.append(plant_folder)



    print("[INFO] Image loading completed from all directories....")  
except Exception as e:
    print("Try...Catch: Error when loding/processing images...: "+str(e))

#Get Size of Processed Image
Tot_No_Of_Images = len(image_list) 
print("Total No. of Images :->"+str(Tot_No_Of_Images))

#Convert labels to numeric values (Ex. 0,1,2,3...based on categories)
le=LabelEncoder()
label_list_num=le.fit_transform(label_list)
label_classes = le.classes_

#print label_list_num
print("Total number of unique categores...="+str(len(np.unique(label_list_num))))
print(np.unique(label_list_num))
n_classes=len(np.unique(label_list_num))

#convert to binary values (one hot encoding)
label_list_num_bin=to_categorical(label_list_num)

np_image_list = np.array(image_list, dtype=np.float16) / 255.0

#Create Train and Test set
print("[INFO] Spliting data to train, test")
#x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 

x_train, x_test, y_train, y_test = train_test_split(np_image_list, label_list_num_bin, test_size=0.2, random_state = 42) 

#Image Augmentation
aug = ImageDataGenerator(
    rotation_range=25, 
    width_shift_range=0.1,
    height_shift_range=0.1, 
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True, 
    fill_mode="nearest")


#Build CNN layers
model = Sequential()

#width=256, height=256, depth=3

inputShape = (height, width, depth)

print("Image data format...")
print(K.image_data_format())

chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1

model.add(Conv2D(32, (3, 3), input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#Fully Connected Layer
model.add(Dense(128))
model.add(Activation("relu"))

#Output layer
model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.summary()

opt = Adam(lr=INIT_LR)

# distribution...Complile the model
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#Plot the train and val curve
#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.figure()

#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


#Model Accuracy
print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print("Printing scores...")
print(scores)
print("Test Loss: "+str(scores[0]*100))
print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("Test Accuracy: "+str(scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Save the model to disk
print("[INFO] Saving model...")
model.save("cnn_model.h5")

