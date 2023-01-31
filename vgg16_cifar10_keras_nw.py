
import pandas as pd
import numpy as np
import time
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1.6, rc={"figure.dpi":250, 'savefig.dpi':250})

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import cifar100, cifar10
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.generic_utils import get_custom_objects
from keras.activations import swish, leaky_relu, relu, gelu, selu, softplus, softsign, elu
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers

import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.optimizers import SGD

import wandb
from wandb.keras import WandbCallback

img_shape = (32, 32, 3)
n_classes = 10

 # load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Scale the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)   

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'dropout': {
            'values': [0.3]
        },
        'epochs': {
            'value': 10
        },
        'batch_size': {
            'values': [64]
        },
        'act': {
            'values': ["relu", "mish", 'esh']
        },
        'learning_rate': {
            'values': [0.01]
        }

    }
}

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Define the Esh activation function
def esh(x):
    return x * K.tanh(K.sigmoid(x))

def mish(x):
    return tfa.activations.mish(x)





def build_model(weight_decay=0.0005, act_func = 'relu'):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=img_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation(act_func))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model

def setup():
    set_seed()
    get_custom_objects().update({'mish': Activation(mish)})
    get_custom_objects().update({'esh': Activation(esh)})

def get_optimizer(lr=1e-3, optimizer="adam"):
    "Select optmizer between adam and sgd with momentum"
    if optimizer.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if optimizer.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.1)

def train(model, batch_size=64, epochs=10, lr=1e-3, optimizer='adam', log_freq=10):  
    
    # Compile model like you usually do.
    tf.keras.backend.clear_session()
    model.compile(loss="categorical_crossentropy", 
                  optimizer=get_optimizer(lr, optimizer), 
                  metrics=["accuracy"])

    # callback setup
    wandb_callbacks = [WandbCallback(compute_flops=True)]
    

    model.fit(x_train, 
              y_train, 
              batch_size=batch_size, 
              epochs=epochs, 
              validation_data=(x_test, y_test), 
              callbacks=wandb_callbacks)

def sweep_train(config_defaults=None):

    with wandb.init(config=config_defaults):  

        wandb.config.architecture_name = "VGG16"
        wandb.config.dataset_name = "CIFAR10"

        # initialize model
        model = build_model(weight_decay=0.0005, act_func=wandb.config.act)

        train(model, 
              wandb.config.batch_size, 
              wandb.config.epochs,
              wandb.config.learning_rate,
              wandb.config.optimizer)

def main():

    #settings
    setup()
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="Esh_vgg16_cifar10_keras")
    wandb.agent(sweep_id, function=sweep_train, count=10)
    

if __name__ == "__main__":

    main()
