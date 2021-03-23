import random
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler,PolynomialCountSketch
import pandas as pd
import sklearn.pipeline
import sklearn.preprocessing
import pickle

class Agent:
    def __init__(self):
        self.prev_action=0
        self.prev_state=np.zeros(9)
        samples_df=pd.read_csv('./samples.csv')
        sample_states=samples_df[["s0","s1","s2","s3","s4","s5","s6","s7","s8"]].values/10
        
        self.featurizer= sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100)),
        ('plsk1',PolynomialCountSketch(degree=3, random_state=1))
        ])
        self.featurizer.fit(sample_states)
        self.models = []
        for i in range(2):
            # self.models.append(pickle.load(open('./model_{i}.sav'.format(i=i), 'rb')))
            
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state([np.zeros(9)])], [0])
            self.models.append(model)
    def featurize_state(self, state):

        scaled = np.array(state)/10
        scaled=scaled.reshape(-1)
        featurized = self.featurizer.transform([scaled])
        return featurized[0]
    def predict(self, s, a=None):
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):

        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])
    def save(self):
        for idx,m in enumerate(self.models):
            pickle.dump(m, open('./model_{f}'.format(f=idx), 'wb'))
    def select_action(self, state, conn=None, vehicle_ids=None):
        features = self.featurize_state(state)

        return np.argmax([m.predict([features])[0] for m in self.models])
