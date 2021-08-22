import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing import image

import os
import random


def orthogonalRot(image):
    # Rotate180 (2), 90CW/270 (3) or 90CCW
    return np.rot90(image, np.random.choice([3, 0, 2]))


# settings for reproducibility
SEED = 42
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# === CONFIGURATION ===
# list of labels
# catergorical data to be converted to numerical data via one-hot encoding
# SWARM_LABELS = ["swirl", "snake"]
# Generation resolution - Must be square
# Training data is also scaled to this.
GENERATE_SQUARE = 256
IMAGE_CHANNELS = 1

# epochs, batch size and location of dataset
DATA_PATH = "data/swarming"  # '/content/drive/My Drive/research/deep_learning/GDL_code/data/bacteria/'
EPOCHS = 75
BATCH_SIZE = 16

print(f"Will generate {GENERATE_SQUARE}px square images.")

# run params
SECTION = 'cnn'
RUN_ID = '0000'
DATA_NAME = 'swarming'
MODEL_FOLDER = 'models/{}/'.format(SECTION)
MODEL_FOLDER += '_'.join([RUN_ID, DATA_NAME])  # where to save the models
print(MODEL_FOLDER)

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
    os.mkdir(os.path.join(MODEL_FOLDER, 'weights'))

# Let's have Keras resize all the images to GENERATE_SQUARE by GENERATE_SQUARE once they've been manipulated.
# width,height,channels
image_shape = (GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)

# === PREPARE THE TRAIING DATA ===
# perform data augmentation to expand the size of the dataset
# https://stackoverflow.com/questions/51748514/does-imagedatagenerator-add-more-images-to-my-dataset
train_gen = ImageDataGenerator(rotation_range=10,  # rotate the image 10 degrees
                               width_shift_range=0.1,  # Shift the pic width by a max of 10%
                               height_shift_range=0.1,  # Shift the pic height by a max of 10%
                               rescale=1/255,  # Rescale the image by normalzing it.
                               shear_range=0.05,  # Shear means cutting away part of the image (max 5%)
                               zoom_range=0.05,  # Zoom in by 5% max
                               brightness_range=[0.75, 1.25],  # alter brightness
                               horizontal_flip=True,  # Allow horizontal flipping
                               vertical_flip=True,  # Allow vertical flipping
                               preprocessing_function=orthogonalRot,
                               fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                               )

# Use flow_from_directory, to set up train and text images
# The directories should only contain images of one class
# so one folder per class of images (e.g. 'train/snakes', 'train/swirls', etc).
train_dir = './data/' + DATA_NAME + '/train'

# generate batches of train images and labels
train_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=image_shape[:2],
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    seed=SEED,
    class_mode='binary',
    shuffle=False)

train_generator.class_indices

# === VISUALIZE THE TRAINING DATASET ===
# choose the image index for the visualization
image_id = 0

# get the train image shape
print("The shape of train images: {}".format(train_generator[image_id][0][0].shape))

# visualize the image example
plt.axis('off')
plt.imshow(train_generator[image_id][0][0])

# get image class and map its index with the names of the classes
train_image_label_id = np.argmax(train_generator[image_id][1][0])
classes_list = list(train_generator.class_indices.keys())

# show image class
plt.title("Class name: {}".format(classes_list[train_image_label_id]))

# === PREPARE THE TEST/VALIDATION DATA ===
# generate batches of validation images and labels
test_gen = ImageDataGenerator(rescale=1/255)

test_dir = './data/' + DATA_NAME + '/test'

validation_generator = test_gen.flow_from_directory(
    test_dir,
    target_size=image_shape[:2],
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    seed=SEED,
    class_mode='binary',
    shuffle=False)

# === VISUALIZE THE TEST/VALIDATION DATASET ===
# choose the image index for the visualization
val_image_id = 0

# get the validation image shape
print("The shape of validation images: {}".format(validation_generator[val_image_id][0][0].shape))

# visualize the image example
plt.axis('off')
plt.imshow(validation_generator[val_image_id][0][0])

# get image class and map its index with the names of the classes
val_image_label_id = np.argmax(validation_generator[val_image_id][1][0])
classes_list = list(validation_generator.class_indices.keys())

# show image class
plt.title("Class name: {}".format(classes_list[val_image_label_id]))

# === CREATE THE MODEL ===
weight_init = RandomNormal(mean=0., stddev=0.02)
batch_norm_momentum = 0.9

input_layer = Input(shape=(GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS), name='model_input')

x = input_layer

x = Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    strides=2,
    padding='same',
    kernel_initializer=weight_init)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization(momentum=batch_norm_momentum)(x)

x = Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu',
    strides=2,
    padding='same',
    kernel_initializer=weight_init)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization(momentum=batch_norm_momentum)(x)

x = Conv2D(
    filters=96,
    kernel_size=(3, 3),
    activation='relu',
    strides=2,
    padding='same',
    kernel_initializer=weight_init)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization(momentum=batch_norm_momentum)(x)

x = Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu',
    strides=2,
    padding='same',
    kernel_initializer=weight_init)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization(momentum=batch_norm_momentum)(x)
x = Dropout(rate=0.25)(x)

x = Flatten()(x)
x = Dense(256, activation='relu', kernel_initializer=weight_init)(x)
x = Dropout(rate=0.4)(x)

x = Dense(128, activation='relu', kernel_initializer=weight_init)(x)

output_layer = Dense(1, activation='sigmoid', kernel_initializer=weight_init)(x)

model = Model(input_layer, output_layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# === TRAIN THE MODEL ===
results = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

# === EVALUATE THE MODEL ===
results.history['loss']

plt.plot(results.history['loss'])

# === PREDICTION ===
snake = test_dir + '/snakes/07_2021-08-17_114155.jpg'
snake_img = image.load_img(snake, target_size=(GENERATE_SQUARE, GENERATE_SQUARE), color_mode="grayscale")
snake_img = image.img_to_array(snake_img)
snake_img = np.expand_dims(snake_img, axis=0)
snake_img = snake_img/255
prediction_prob = model.predict(snake_img)

# Output prediction
print(f'Probability that the image is a snake: {1 - prediction_prob[0][0]} ')

swirl = test_dir + '/swirls/20_2021-08-17_180217.jpg'
swirl_img = image.load_img(swirl, target_size=(GENERATE_SQUARE, GENERATE_SQUARE), color_mode="grayscale")
swirl_img = image.img_to_array(swirl_img)
swirl_img = np.expand_dims(swirl_img, axis=0)
swirl_img = swirl_img/255
prediction_prob = model.predict(swirl_img)

# Output prediction
print(f'Probability that the image is a swirl: {prediction_prob} ')
