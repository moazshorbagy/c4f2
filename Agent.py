import random
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler,PolynomialCountSketch
import pandas as pd
import sklearn.pipeline
import sklearn.preprocessing
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Agent:
    def __init__(self):    
        num_inputs = 22
        num_actions = 2
        num_hidden = 256

        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden*2, activation="relu")(inputs)
        common2 = layers.Dense(num_hidden, activation="relu")(common)
        common3 = layers.Dense(num_hidden, activation="relu")(common2)

        action = layers.Dense(num_actions, activation="softmax")(common3)
        critic = layers.Dense(1)(common3)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])
    def select_action(self, state, conn=None, vehicle_ids=None):
        features = self.featurize_state(state)

        return np.argmax([m.predict([features])[0] for m in self.models])
