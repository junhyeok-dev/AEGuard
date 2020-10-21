import os
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
import pandas as pd

os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(10)

random_dim = 100

def get_optimizer():
    return Adam(learning_rate=0.001, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add()