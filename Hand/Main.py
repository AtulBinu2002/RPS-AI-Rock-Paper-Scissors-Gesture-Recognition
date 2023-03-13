import random
import os
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import pathlib
import matplotlib.image as mpimg 

#Path
train_dir = "D:/Visual/Project/Image_Classification/Data/Hand_1/train/"
val_dir="D:/Visual/Project/Image_Classification/Data/Hand_1/val/"
test_dir = "D:/Visual/Project/Image_Classification/Data/Hand_1/test/"
model_dir="D:/Visual/Project/Image_Classification/Hand/Final.h5"

data_dir =pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

# Setup target directory (we'll view images from here)
target_folder ="D:/Visual/Project/Image_Classification/Data/Hand_1/test/"+random.sample(["rock","scissors","paper"],1)[0]

model_baseline=tf.keras.models.load_model(model_dir)

# Get a random image path
random_image = random.sample(os.listdir(target_folder), 1)

# Read in the image and plot it using matplotlib
image = tf.io.read_file(target_folder + "/" + random_image[0])

img = tf.image.decode_image(image, channels=3)
img = tf.image.resize(img, size = [224,224])/255

pred = model_baseline.predict(tf.expand_dims(img, axis=0))

# Get the predicted class
pred_class = class_names[pred.argmax()]
print(pred_class,"Accuracy=",pred.max()*100)
