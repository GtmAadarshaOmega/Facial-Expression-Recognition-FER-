#Part 1: Importing Required Libraries and Modules
#In this part, we import the necessary libraries for building and training the model. TensorFlow is the deep learning framework we'll use. We import specific modules from Keras to create the model layers.


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


#Part 2: Loading and Preprocessing the Dataset
#In this part, we load and preprocess the dataset using ImageDataGenerator. We specify augmentation and normalization transformations for the training data. 
#The generators (train_generator and validation_generator) will provide batches of preprocessed images and labels during training.


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'path/to/train/dataset'
validation_data_dir = 'path/to/validation/dataset'
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

#Part 3: Creating the CNN Model
#Here, we define the architecture of the Convolutional Neural Network (CNN) model. 
#The model consists of convolutional layers, max-pooling layers, flattening layer, and fully connected layers. 
#The final output layer uses softmax activation for multi-class classification.

input_shape = (224, 224, 3)
num_classes = 7

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
