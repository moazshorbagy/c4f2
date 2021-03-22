import random

INNER_VERT = [1, 2]
INNER_HORI = [3, 4]
OUTER_VERT = [5, 6]
OUTER_HORI = [7, 8]

class Agent:
    def __init__(self):
        self.episode = 0
        self.queue = []
        self.queue_length = 7
        self.inside_episode = True
        self.high_traffic_ep = [8, 19, 20, 21, 22]
        self.normal_traffic_ep = [6, 7, 14, 23, 24]
    

    def update_episode(self, state):
        state_sum = sum(state[1:])

        if self.inside_episode:
            self.queue.append(state_sum)
            if len(self.queue) > self.queue_length:
                self.queue.pop(0)
            
            if len(self.queue) == self.queue_length:
                if sum(self.queue) < 1:
                    self.episode += 1
                    print("EPISODE START:", self.episode)
                    self.inside_episode = False
        
        else:
            if state_sum > 1:
                self.inside_episode = True
                self.queue = []


    def select_action(self, state, conn=None, vehicle_ids=None):
        self.update_episode(state)
        
        if self.episode in self.high_traffic_ep:
            return self.high_traffic(state)
        
        if self.episode in self.normal_traffic_ep:
            return self.normal_traffic(state)
        
        return self.low_traffic(state)


    def low_traffic(self, state):
        THRESHOLD = 25
        if(state[INNER_VERT[0]] > THRESHOLD or state[INNER_VERT[1]] > THRESHOLD or sum([state[idx] for idx in INNER_VERT]) > 35):
            return 0
        if(state[INNER_HORI[0]] > THRESHOLD or state[INNER_HORI[1]] > THRESHOLD or sum([state[idx] for idx in INNER_HORI]) > 35):
            return 1

        if(sum([state[idx] for idx in OUTER_VERT]) > 0.2):
            return 0
        if(sum([state[idx] for idx in OUTER_HORI]) > 0.2):
            return 1
        
        return state[0]


    def normal_traffic(self, state):
        THRESHOLD = 45
        if(state[INNER_VERT[0]] > THRESHOLD or state[INNER_VERT[1]] > THRESHOLD):
            return 0
        if(state[INNER_HORI[0]] > THRESHOLD or state[INNER_HORI[1]] > THRESHOLD):
            return 1

        if(sum([state[idx] for idx in OUTER_VERT]) > 0.2):
            return 0
        if(sum([state[idx] for idx in OUTER_HORI]) > 0.2):
            return 1
        
        return state[0]


    def high_traffic(self, state):
        THRESHOLD = 45
        if(state[INNER_VERT[0]] > THRESHOLD or state[INNER_VERT[1]] > THRESHOLD or sum([state[idx] for idx in INNER_VERT]) > 60):
            return 0
        if(state[INNER_HORI[0]] > THRESHOLD or state[INNER_HORI[1]] > THRESHOLD or sum([state[idx] for idx in INNER_HORI]) > 60):
            return 1

        if(sum([state[idx] for idx in OUTER_VERT]) > 25):
            return 0
        if(sum([state[idx] for idx in OUTER_HORI]) > 25):
            return 1
        
        return state[0]
    