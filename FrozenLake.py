# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:05:05 2021

@author: s161981
"""
import numpy as np
import gym
from matplotlib import pyplot as plt 


###TODO###
# Initialisatie van V en Q functies/arrays zijn nu nog 1 voor goal, 0 voor de rest. Geen idee of dat klopt
# Error plots zijn alleen nog maar constant 0. Bij TD leek er wel iets op convergence toen ik een andere methode dan in het boek gebruikte, maar geen idee of en waarom die klopt

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
    global actions, nrStates, final, returns, state, V, Q, env
    actions = ['left', 'down', 'right', 'up']
    nrStates = size**2
    state = 0
    final = nrStates - 1
    returns = np.zeros(nrStates)
    
    #for TD: 
    V = np.zeros(nrStates) #TODO probably initialise this inside TD, not here
    V[final] = 1
    
    #for Q-learning:
    Q = np.zeros((nrStates, 4)) # TODO should we initialise to zero?
    Q[final,:] = 1
    
    if size == 8:
        env = gym.make ("FrozenLake8x8-v0")
    else:
        env = gym.make ("FrozenLake-v0")
        



### TD learning ###
#after Sutton & Barto, p120 

#TODO: maybe implement TD, Qlearning etc. differently to re-use parts of the loop below?
# benefit of current approach is ease of comparison to pseudocode in book.

def TD(nrEpisodes, alpha, gamma, policy = None, printSteps=True): 
    ''' Runs TD(0) algorithm for given number of episodes to estimate value function V '''
    
    errPerState = {i:list() for i in range(nrStates)} # keeps track of error in each state.
    # (note that tracking all errors in one list does not make sense, since some states
    #  are visited more often earlier on and thus already show some convergence.)
    for n in range(nrEpisodes):
        env.reset() # Reset environment every episode?
        state = 0
        t = 0
        # Run the game
        while True:
            if printSteps: env.render()
            
            # Choose action 
            if policy == None:
                action = env.action_space.sample() #choose random action
            else: 
                action = policy[state] #follow policy
            
            # Take chosen action, visit new state and obtain reward
            newState, reward, done, info = env.step(action)
            
            # Update V and keep track of errors:
            err = reward + gamma * V[newState] - V[state]
            V[state] += alpha * err
            errPerState[state].append(err) # TODO: now all negative... but at least converging
            #not too happy with appending, but cannot know in advance how long it will become
            #also, is the indexing correct? Book mentions this error is not available until next timestep (below eq 6.5)
            
            #TODO: sometimes, done like this: see e.g. https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b
            # old = V[state] # must be stored here, for the case state = newState
            # V[newState] += alpha * (reward + gamma * V[final] - V[newState])
            
            # # Keep track of errors: now all negative... but at least converging
            # TDerrPerState[state].append(reward + gamma*V[newState] - old)
                
            state = newState
            t += 1
            
            if printSteps: 
                print("At time", t, ", we obtained reward", reward, ", and visited:", newState, "\n")
                # print("Next action:", actions[action])
            
            if done:
                if printSteps: print("Episode finished" )
                break
        
        env.close()
    return V, errPerState


### Q-learning: ###
#after Sutton & Barto, p131

def Qlearning(nrEpisodes, alpha, gamma, epsilon, printSteps=True):
    errPerState = {i:list() for i in range(nrStates)} # keeps track of error in each state.
    for n in range(nrEpisodes):
        env.reset() # Reset environment every episode?
        state = 0
        t = 0
        
        # Run the game
        while True:
            if printSteps: env.render()
            
            #Choose action using eps-greedy policy from Q
            if np.random.rand() >= epsilon: # with probability 1-epsilon, choose greedily
                action = np.argmax(Q[state,:]) 
            else: # with probability epsilon, do not choose greedy
                action = env.action_space.sample() #chooses random action
            
            # Take chosen action, visit new state and obtain reward
            newState, reward, done, info = env.step(action)
            
            # Update Q and save error to state:
            err = reward + gamma * np.max(Q[newState,:]) - Q[state, action]
            Q[state, action] += alpha * err
            errPerState[state].append(err)
                
            state = newState
            t += 1
            
            if printSteps: 
                print("At time", t, ", we obtained reward", reward, ", and visited:", newState, "\n")
                # print("Next action:", actions[action])
            
            if done:
                if printSteps: print("Episode finished" )
                break
        
        env.close()
    return Q, errPerState



### SARSA: ###
#after Sutton & Barto, p130

def SARSA(nrEpisodes, alpha, gamma, epsilon, printSteps=True):
    errPerStateAction = {(i,a):list() for i in range(nrStates) for a in range(4)} # keeps track of error in each state.
    for n in range(nrEpisodes):
        env.reset() # Reset environment every episode?
        t = 0
        
        #state-action initialisation (action choice is epsilon-greedy):
        state = 0
        if np.random.rand() >= epsilon: 
            action = np.argmax(Q[state,:]) 
        else: 
            action = env.action_space.sample()
        
        # Run the game
        while True:
            if printSteps: env.render()
            
            newState, reward, done, info = env.step(action)
            
            #Choose action using eps-greedy policy from Q
            if np.random.rand() >= epsilon: # with probability 1-epsilon, choose greedily
                newAction = np.argmax(Q[newState,:]) 
            else: # with probability epsilon, do not choose greedy
                newAction = env.action_space.sample() #chooses random action
            
            # Take chosen action, visit new state and obtain reward
            
            # Update Q and save error to state:
            err = reward + gamma * Q[newState,newAction] - Q[state, action]
            Q[state, action] += alpha * err
            errPerStateAction[state,action].append(err)
                
            state = newState
            action = newAction
            t += 1
            
            if printSteps: 
                print("At time", t, ", we obtained reward", reward, ", and visited:", newState, "\n")
                # print("Next action:", actions[action])
            
            if done:
                if printSteps: print("Episode finished" )
                break
        
        env.close()
    return Q, errPerStateAction




### Plot error ###
def plotFromDict(errorDict): 
    plt.clf() # Clears current figure
    plt.rcParams.update({'font.size': 12})
    
    # Errors per state: (inspired by https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b)
    errorLists = [list(x)[:50] for x in errorDict.values()]
    for errorList in errorLists:
        plt.plot(errorList)
    
    plt.xlabel('Number of Visits')
    plt.ylabel('Error')
    plt.show()
    plt.savefig("FrozenLake.pdf", bbox_inches = 'tight')



### Execution ###
createEnv()

nrEpisodes = 1000
alpha = 0.1 #Stepsize
gamma = 0.1 #Discounting rate; may differ per state and iteration

# TD: 
# arbitraryPolicy = np.random.choice(range(4),nrStates) #randomly chosen policy, but fixed from now on
# fixedPolicy = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2] #fixed policy

# V, errors = TD(nrEpisodes, alpha, gamma, printSteps = False) #optional arguments: policy and printSteps

# Q-learning:
epsilon = 0.1
#Q, errors = Qlearning(nrEpisodes, alpha, gamma, epsilon, printSteps = False)


# SARSA:
epsilon = 0.1
Q, errors = SARSA(nrEpisodes, alpha, gamma, epsilon, printSteps = False)

plotFromDict(errors)




### Currently unused: ###

#Could also opt for actual square representation?
def getCoordinate(index, gridsize):
    '''Takes location index and turns it into grid coordinate '''
    return (index // gridsize, index % gridsize) # // is floor division

# V = np.zeros((4,4)) #book says to initialise arbitrarily except for terminal states (for which V is 0)
# returns = np.zeros((4,4))
# d = np.zeros((4,4))