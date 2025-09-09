import numpy as np

class World:
    def __init__(self, size):

        self.grid = np.zeros((size, size, 3))


    def update(self, obs):

        image = obs['image']
        orientation = obs['orientation']

        # if orientation == 0:
        #     self.grid[] = image


        
