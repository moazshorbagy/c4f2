import random

INNER_VERT = [1, 2]
INNER_HORI = [3, 4]
OUTER_VERT = [5, 6]
OUTER_HORI = [7, 8]
THRESHOLD = 45

class Agent:
    def select_action(self, state, conn=None, vehicle_ids=None):
        if(state[INNER_VERT[0]] > THRESHOLD or state[INNER_VERT[1]] > THRESHOLD):
            return 0
        if(state[INNER_HORI[0]] > THRESHOLD or state[INNER_HORI[1]] > THRESHOLD):
            return 1

        if(sum([state[idx] for idx in OUTER_VERT]) > 0.2):
            return 0
        if(sum([state[idx] for idx in OUTER_HORI]) > 0.2):
            return 1
        
        return state[0]
