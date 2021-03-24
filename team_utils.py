import numpy as np

class State_Buffer():
    def __init__(self,size,n_features):
        self.cntr=0
        self.size=size
        self.n_features=n_features
        self.states=np.zeros((size,9)).tolist()
    def get_states(self):
        buffered=[]
        for s in self.states:
                close1=s[1]+s[2]
                close2=s[3]+s[4]+.1
                far1=s[5]+s[6]
                far2=s[7]+s[8]+.1
                buffered.append([s[0],s[1]-s[5],s[2]-s[6],s[3]-s[7],s[4]-s[8],close1,close2,far1,far2,close1/close2,far1/far2])
        return np.array(buffered).reshape(self.size*self.n_features)
    def add_state(self,s):
        self.states.append(s)
        self.states.pop(0)