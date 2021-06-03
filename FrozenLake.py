# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:05:05 2021

@author: s161981
"""
import numpy as np
import gym

def createEnv(size = 4): 
    '''
    Creates frozen lake game environment in 8x8 if unspecified, else 4x4
    See https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
    '''
    if size == 8:
        env = gym.make ("FrozenLake8x8-v0")
    else:
        env = gym.make ("FrozenLake-v0")
    
    return env


env = createEnv()

# print(env.action_space) # Up, down, right, left
# print(env.observation_space) # 8x8 grid

# print("Press Enter to continue..." )
# input()


def getCoordinate(index, gridsize):
    '''Takes location index and turns it into grid coordinate '''
    return (index // gridsize, index % gridsize) # // is floor division


### Temporal Difference ###
#after Sutton & Barto, p120 
nrSteps = 1000
nrEpisodes = 3
gamma = 1 #gamma, in slides. May differ per state and iteration
alpha = 1

V = np.zeros(16)
final = 15
V[final] = 1
returns = np.zeros(16)
d = np.zeros(16)
state = 0

# Locations are numbered from left to right, then top to bottom, 
# e.g., in 4x4 version, 
#   0  1  2  3
#   4  5  6  7
#   8  9  10 11
#   12 13 14 15
# but are stored in a list.

#Could also opt for actual square representation?
# V = np.zeros((4,4)) #book says to initialise arbitrarily except for terminal states (for which V is 0)
# returns = np.zeros((4,4))
# d = np.zeros((4,4))

policy = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2] #quite arbitrary, just for testing

for n in range(nrEpisodes):
    env.reset() # Reset environment every episode?
    
    # Run the game
    for t in range(nrSteps):
        env.render()
        
        # Choose a random action #IMPLEMENT CHOICE RULE HERE
        #action = env.action_space.sample()
        action = policy[state]
        
        # Take chosen action, visit new state and obtain reward
        newState, reward, done, info = env.step(action)
        
        
        # Update V:
        V[newState] += alpha * (reward + gamma * V[final] - V[newState])
        
        state = newState
        
        print ("At time " , t , ", we obtained reward " , reward, ", and visited: ", newState)
        print(action)
        
        if done:
                print ("Episode finished" )
                break
    
    env.close()
print(V)    