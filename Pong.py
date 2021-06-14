# -*- coding: utf-8 -*-
"""
@author: s161981, s1017389
"""
import numpy as np
import gym
from matplotlib import pyplot as plt 
np.random.seed(1964)

### Initialise game settings etc.
def createEnv(): 
    '''
    Creates pong environment from gym package
    '''
    global nrStates, nrActions, startState, finalState, returns, env
    env = gym.make ("Pong-v0")
    # env.action_space # Discrete(6)
    # env.observation_space # Box(0, 255, (210, 160, 3), uint8)


def policyEvaluation(nrEpisodes, printSteps):
    # Run game nrEpisodes times
    for n in range(nrEpisodes):
        currentState = env.reset() # reset game to initial state
        done = False
        t = 0
        
        # Run one game
        while not(done):
            if printSteps: 
                env.render()
            action = env.action_space.sample()
            
            # Take chosen action, visit new state and obtain reward
            newState, reward, done, info = env.step(action)
            
            # Update state and time
            currentState = newState
            t += 1
        env.close()
        
### Execution ###
createEnv() # create game 

nrEpisodes = 1
printSteps = True
alpha = .02 # stepsize
gamma = 1 # discounting rate; may differ per state and iteration
eps = .9 # initial value
decay_rate = .5
min_epsilon = 0.01
progressPoints = 100

policyEvaluation(nrEpisodes, printSteps)




