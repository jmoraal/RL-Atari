# -*- coding: utf-8 -*-
"""
@author: s161981, s1017389
"""
import numpy as np
import gym
from matplotlib import pyplot as plt 
import scipy.stats as st
from skimage.transform import resize
np.random.seed(1964)

### Initialize game settings etc.
def createEnv(): 
    ''' Create game environment from Gym package. '''
    global nrStates, nrActions, env
    
    env = gym.make ("Pong-v0")
    nrStates = 2**(25)
    nrActions = 6

def greedy(Q, state):
    ''' Makes greedy choice for action given state and value table. '''
    argmaxes = np.flatnonzero(Q[state,:] == np.max(Q[state,:]))
    return np.random.choice(argmaxes)

def epsGreedy(Q, state, epsilon = 0.05): 
    ''' Makes epsilon-greedy choice for action given state and value table. '''
    if np.random.rand() > epsilon: # with probability 1-epsilon, choose current best option greedily
        return greedy(Q,state)
    else: # with probability epsilon, choose randomly
        return env.action_space.sample()

# Quality check of given policy
def policyPerformanceStats(Q, policy = greedy, nrGames = 1): 
    '''Performs same given policy over and over to measure accuracy, 
    outputs mean, std and confidence interval of mean
    
    2000 games is enough for an indication of progress; much more would 
    slow down the iteration loop too much. 
    
    For narrow confidence interval to evaluate final policy, 
    set nrGames to higher value '''
    
    rewards = np.zeros(nrGames)
    
    for i in range(nrGames):
        done = False
        state = env.reset()
        state  = reduceState(state)
        gameReward = 0
        
        while not done: 
            env.render()
            action = policy(Q, state)
            state, reward, done, info = env.step(action)
            state  = reduceState(state)
            gameReward += reward
        
        rewards[i] = gameReward
        
    env.close()
    
    return summaryStats(rewards)

def summaryStats(data, confidence = 0.95): 
    mean = np.mean(data)
    std = np.std(data)
    confInt = st.t.interval(0.95, len(data)-1, loc=mean, scale=st.sem(data)) 
    
    return mean, std, confInt

def reduceState(state): 
    ''' Reduces dimension of state. '''
    state = state[34:194,:,:] # remove top 34 and bottom 16 pixels
    state = state[:,:,0] # remove third axis (color channels)
    state = np.where(state == 144, 0, state) # brown is background so set those values to 0
    
    state = np.where(state != 0, 255, state) # set all other values (green, white, orange) to 255
    
    state = resize(state, (5, 5)) # downscale image
    state = np.where(state != 0, 1, 0)
    state = state.flatten()
    state = state.dot(2**np.arange(state.size)[::-1]) # convert binary array to integer (=number of state) using dot product

    return int(state)
    
# Main function: policy evaluation/improvement                    
def policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod , epsilon = 0, printSteps = True, VfromFile = False): 
    global V, gameDurations, pointsLost, pointsWon
    
    gameDurations = [] # for analysis
    pointsLost = np.zeros(nrEpisodes)  
    pointsWon = np.zeros(nrEpisodes)  
    
    # Initialize value function and error lists
    if evaluationMethod == "Q":
        if VfromFile:
            V = np.load(evaluationMethod+"V"+".npy")
        else:
            V = np.ones((nrStates,nrActions)) 
    elif evaluationMethod == "SARSA":
        if VfromFile:
            V = np.load(evaluationMethod+"V"+".npy")
        else:
            V = np.ones((nrStates,nrActions))
        action = env.action_space.sample() 

        
    # Run game nrEpisodes times
    for n in range(nrEpisodes):
        currentState = env.reset() # reset game to initial state
        currentState = reduceState(currentState)
        
        done = False
        t = 0
        lost = 0
        won = 0
    
        # Run one game
        while not(done):
            if printSteps: 
                env.render()
                
            # Evaluate policy
            #Q-learning:
            if evaluationMethod == "Q":
                action = epsGreedy(V, currentState, epsilon = epsilon)
                
                newState, reward, done, info = env.step(action)
                newState  = reduceState(newState)
                
                
                tempValue = V[currentState, action]
                tempMax = np.max(V[newState,:])
                V[currentState,action] += alpha*(reward + gamma*tempMax - tempValue)
                
            #SARSA:
            elif evaluationMethod == "SARSA":
                newState, reward, done, info = env.step(action)
                
                newState  = reduceState(newState)
                
                newAction = epsGreedy(V, newState, epsilon = epsilon)
                
                tempValue = V[currentState, action]
                V[currentState,action] += alpha*(reward + gamma*V[newState,newAction] - V[currentState, action])
                action = newAction 
                
            # Keeping score
            if reward == -1: lost += 1
            if reward == 1: won += 1
            
            # Update state
            currentState = newState
            t += 1
            
        
        print(f"Episode {n} finished after {t+1} timesteps with epsilon {epsilon} and points {won}")
        gameDurations.append(t+1) # won the game
        pointsLost[n] = lost
        pointsWon[n] = won
        
        epsilon *= decay_rate
        epsilon = max(epsilon, min_epsilon)
    
    return V, gameDurations, np.asarray(pointsLost), np.asarray(pointsWon)


### Analysis ###
def plotInitialize():
    plt.clf() # Clears current figure
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 12})

# Summary
def printGameSummary(durations, pointsLost, pointsWon, evaluationMethod):
    print(evaluationMethod,":")
    print(f"Mean points lost: {np.mean(pointsLost)}")
    print(f"Mean points won: {np.mean(pointsWon)}")
    print(f"Mean game duration: {np.mean(durations)}")
    
    print(f"Mean points lost over last 100 games: {np.mean(pointsLost[-100:len(pointsLost)])}")
    print(f"Mean points won over last 100 games: {np.mean(pointsWon[-100:len(pointsWon)])}")
    print(f"Mean game duration over last 100 games: {np.mean(durations[-100:len(durations)])}")
    print(f"Number of states visited: {np.count_nonzero(V != 1) }")

def plotPoints(pointsLost, pointsWon, title):
    plotInitialize()

    plt.plot(pointsLost, label = "lost")
    plt.plot(pointsWon, label = "won")
    
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.title('Points')
    plt.legend()#prop={'size': 16})
    plt.savefig("Pong-"+title+".pdf", bbox_inches = 'tight')
    # plt.show()
    
def saveOutput(V, durations, pointsLost, pointsWon, title): 
    np.save(title+"V", V)
    np.save(title+"durations", durations)
    np.save(title+"pointsLost", pointsLost)
    np.save(title+"pointsWon", pointsWon)
            

### Execution ###
createEnv() # create game 

nrEpisodes = 500
alpha = .02 # stepsize
gamma = .5 # discounting rate; may differ per state and iteration
eps = .9 # initial value
decay_rate = .9
min_epsilon = 0.05

def runSimulation(evaluationMethod):
    global V, valueUpdates, errors, durations, gamesWon, winRatios, confIntervals
    V, durations, pointsLost, pointsWon = policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod = evaluationMethod, epsilon = eps, printSteps = False, VfromFile = True)
                                                                                               
    printGameSummary(durations, pointsLost, pointsWon, evaluationMethod)
    plotPoints(pointsLost, pointsWon, evaluationMethod)
    statesVisited = np.count_nonzero(V != 1)
    
    return V, durations, pointsLost, pointsWon, statesVisited
    
# Q-learning:
evaluationMethod = "Q"
V, durations, pointsLost, pointsWon, statesVisited = runSimulation(evaluationMethod)

# QsaveV = V
# QsaveDurations = durations
# QsavePointsLost = pointsLost
# QsavePointsWon = pointsWon
# QsaveStatesVisited = statesVisited

# saveOutput(QsaveV,QsaveDurations,QsavePointsLost,QsavePointsWon,valuationMethod)

# SARSA:
evaluationMethod = "SARSA"

V, durations, pointsLost, pointsWon, statesVisited = runSimulation(evaluationMethod)
# SARSAsaveV = V
# SARSAsaveDurations = durations
# SARSAsavePointsLost = pointsLost
# SARSAsavePointsWon = pointsWon
# SARSAsaveStatesVisited = statesVisited

# saveOutput(SARSAsaveV,SARSAsaveDurations,SARSAsavePointsLost,SARSAsavePointsWon,valuationMethod)

# policyPerformanceStats(SARSAsaveV)



            
