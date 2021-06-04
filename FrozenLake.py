# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:05:05 2021

@author: s161981
"""
import numpy as np
import gym
from matplotlib import pyplot as plt 

### Initialise game settings etc.
def createEnv(size = 4): 
    '''
    Creates frozen lake game environment in 4x4 if unspecified, else 8x8
    
    State space: square grid. Locations are numbered from left to right, then 
    top to bottom, and stored in a list. E.g. in 4x4 version:
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15
    Action space: 
        0   left
        1   down
        2   right
        3   up
        
    States can be marked S,H,F,G: start, hole, frozen, goal. 
    Objective is to reach goal (reward 1) from start via frozen tiles; stepping on H means drowning (reward 0).
    See https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py or https://gym.openai.com/envs/FrozenLake-v0/
    '''
    global nrStates, V, final, returns, state, env
    nrStates = size**2
    V = np.zeros(nrStates)
    final = nrStates - 1
    V[final] = 1
    returns = np.zeros(nrStates)
    state = 0
    
    if size == 8:
        env = gym.make ("FrozenLake8x8-v0")
    else:
        env = gym.make ("FrozenLake-v0")



createEnv()



### Temporal Difference Simulation parameters ###
nrEpisodes = 100
alpha = 0.2 #Stepsize
gamma = 0.2 #Discounting rate; may differ per state and iteration


#policy = np.random.choice(range(4),nrStates) #random policy
#policy = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2] #fixed policy


### Simulation with TD ###
#after Sutton & Barto, p120 

TDerrPerState = {i:list() for i in range(nrStates)} # keeps track of error in each state.
# (note that tracking all errors in one list does not make sense, since some states
#  are visited more often earlier on and thus already show some convergence.)


for n in range(nrEpisodes):
    env.reset() # Reset environment every episode?
    state = 0
    t = 0
    # Run the game
    while True:
        env.render()
        
        # Choose action 
        action = env.action_space.sample() #choose random action
        #action = policy[state] #follow policy
        
        # Take chosen action, visit new state and obtain reward
        newState, reward, done, info = env.step(action)
        
        # Update V:
        old = V[state] # must be stored here, for the case state = newState
        V[newState] += alpha * (reward + gamma * V[final] - V[newState])
        
        # Keep track of errors: now all negative... but at least converging
        TDerrPerState[state].append(reward + gamma*V[newState] - old)
        #not too happy with appending, but cannot know in advance how long it will become
        #also, is the indexing correct? Book mentions this error is not available until next timestep (below eq 6.5)
            
        state = newState
        t += 1
        
        print ("At time ", t, ", we obtained reward ", reward, ", and visited: ", newState)
        print(action)
        
        if done:
                print ("Episode finished" )
                break
    
    env.close()
#print(V)


### Plot error ###
plt.clf() # Clears current figure
plt.rcParams.update({'font.size': 12})

# All errors in one sequence:
#plt.plot(TDerrors)

# Errors per state: (inspired by https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b)
errorLists = [list(x)[:50] for x in TDerrPerState.values()]
for errorList in errorLists:
    plt.plot(errorList)
#plt.legend( (,,), loc = 'best') 

plt.xlabel('Error')
plt.ylabel('Iteration')
plt.show()
plt.savefig("FrozenLake.pdf", bbox_inches = 'tight')




### Currently unused: ###

#Could also opt for actual square representation?
def getCoordinate(index, gridsize):
    '''Takes location index and turns it into grid coordinate '''
    return (index // gridsize, index % gridsize) # // is floor division

# V = np.zeros((4,4)) #book says to initialise arbitrarily except for terminal states (for which V is 0)
# returns = np.zeros((4,4))
# d = np.zeros((4,4))