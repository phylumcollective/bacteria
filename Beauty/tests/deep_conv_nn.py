
import numpy as np

from keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

from keras.datasets import cifar10


NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

x_train[54, 12, 13, 1]

input_layer = Input(shape=(32, 32, 3))

conv_layer_1 = Conv2D(
    filters=10,
    kernel_size=(4,4),
    strides=2,
    padding='same'
)(input_layer)

conv_layer_2 = Conv2D(
    filters=20,
    kernel_size=(3,3),
    strides=2,
    padding='same'
)(conv_layer_1)

flatten_layer = Flatten()(conv_layer_2)

output_layer = Dense(uints=10, activation='softmax')(flatten_layer)

model = Model(input_layer, output_layer)

model.summary()


input_layer = Input((32, 32, 3))

x = Conv2D(
    filters=32,
    kenrel_size=3,
    strides=1,
    padding='same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(
    filters=32,
    kernel_size=3,
    strides=2,
    padding='same'
)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(
    filters=64,
    kernel_size=3,
    strides=1,
    padding='same'
)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=0.5)(x)

x = Dense(NUM_CLASSES)(x)
output_layer = Activation('softmax')(x)

model = Model(input_layer, output_layer)

model.summary()

#!!!!!!TRAINING!!!!!!
opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(
    x_train,
    batch_size=32,
    epochs=10,
    shuffle=True,
    validation_data=(x_test, y_test)
)[]

model.layers[6].get_weights()

model.evaluate(x_test, y_test, batch_size=1000)

CLASSES = np.array(['bs7ag0-5pep', 'pd7ag0-5pep',
                    'bs7ag2pep', 'pd7ag2pep',
                    'bs7ag10pep', 'pd7ag10pep',
                    'bs20ag0-5pep', 'pd20ag0-5pep',
                    'bs20ag2pep', 'pd20ag2pep'])

preds = model.predict(x_test)
preds_single = CLASSES(np.argmax(preds, axis = -1))
actual_single = CLASSES(np.argmax(y_test, axis = -1))

import matplotlib.pylot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fonsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)
