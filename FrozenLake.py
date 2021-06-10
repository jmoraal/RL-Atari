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
    Creates frozen lake game environment in 4x4 if unspecified, else 8x8, 
    and initialises variables for TD, Q-learning and SARSA methods. 
    
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
    global nrStates, nrActions, startState, finalState, returns, env
    
    if size == 8:
        env = gym.make ("FrozenLake8x8-v0")
    else:
        env = gym.make ("FrozenLake-v0")
        
    nrStates = env.nS
    nrActions  = env.action_space.n
    startState = 0
    finalState = nrStates - 1
    returns = np.zeros(nrStates)


def epsGreedy(Q, state, epsilon = 0.05): 
    ''' Makes epsilon-greedy choice for action given state and value table'''
    if np.random.rand() > epsilon: # with probability 1-epsilon, choose current best option greedily
        return np.argmax(Q[state,:])
    else: # with probability epsilon, choose randomly
        return env.action_space.sample()
                    
def policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod, epsilon = 0, printSteps = True): 
    
    gameDurations = [] # for analysis
    gamesWon = []
    
    # Initialize value function and error lists
    if evaluationMethod == "TD":
        #V = np.random.rand(nrStates) 
        V = np.ones(nrStates)
        #V = np.zeros(nrStates)
        #V[finalState] = 0
        errors = {i:list() for i in range(nrStates)}  # keeps track of error in each state
    elif evaluationMethod == "Q":
        # V = np.random.rand(nrStates,nrActions)
        V = np.ones((nrStates,nrActions))
        #V[finalState,:] = 0
        errors = {i:list() for i in range(nrStates*nrActions)}  # 2d matrix mapped to vector! J: Why? Doesn't it only make indexing difficult further on
    elif evaluationMethod == "SARSA":
        V = np.random.rand(nrStates,nrActions)
        # V = np.ones((nrStates,nrActions))
        #V[finalState,:] = 0
        errors = {i:list() for i in range(nrStates*nrActions)} 
        #currentState = startState
        action = env.action_space.sample()  #TODO needs to be initialized for SARSA 
                                            #J: but this is fine right?
        
        
    # Run game nrEpisodes times
    for n in range(nrEpisodes):
        currentState = env.reset() # reset game to initial state
        done = False
        t=0
        
        # Run one game
        while not(done):
            if printSteps: 
                env.render()
                
            # Evaluate policy
            #Temporal difference:
            if evaluationMethod == "TD":
                action = env.action_space.sample() #TODO depending on policy?
                
                # Take chosen action, visit new state and obtain reward
                newState, reward, done, info = env.step(action)
                
                # Update value function
                tempValue = V[currentState]
                V[currentState] += alpha*(reward + gamma*V[newState] - V[currentState]) # S: or should we work with a np copy of V (like in tutorial)?
                errors[currentState].append(float(np.abs(tempValue-V[currentState])))
            
            #Q-learning:
            elif evaluationMethod == "Q":
                action = epsGreedy(V, currentState, epsilon = epsilon)
                
                newState, reward, done, info = env.step(action)
            
                tempValue = V[currentState, action]
                a_max = np.argmax(V[newState,:]) # find optimal new action
                V[currentState,:] += alpha*(reward + gamma*V[newState,a_max] - V[currentState, action]) 
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) #TODO S: same errors as TD ??
                #J: In this formulation yes; but errors are not actually equal as they depend on the update. 
                #TODO Also, error now changes with alpha, book does not do this; what should we take?
                
            #SARSA:
            elif evaluationMethod == "SARSA":
                action = epsGreedy(V, currentState, epsilon = epsilon)
                newState, reward, done, info = env.step(action)
                newAction = epsGreedy(V, newState, epsilon = epsilon)
                
                tempValue = V[currentState, action]
                V[currentState,action] += alpha*(reward + gamma*V[newState,newAction] - V[currentState, action])
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) #TODO S: same errors as TD ??
                action = newAction 
              
            # Update state
            currentState = newState
            t += 1
            
        # Print results if desired
        if printSteps:
            directions =  ["L", "D", "R", "U"]
            print("At time", t, ", we obtain reward", reward, ", choose ", directions[action], " and move to:", newState, "\n")
        
        if printSteps: print(f"Episode finished after {t+1} timesteps" )
        if reward == 1: gameDurations.append(t+1) # won the game
        gamesWon.append(reward)
            
            # Update policy using value function
            # Now that we have the value function of all the states, our next step is to extract the policy from the Value Function.
            # See https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
            # policy = updatePolicy(...)
            
            #J: is this still relevant? At least for Q and SARSA, policy is defined by Q (V) and hence implicitly updated
         
#         env.close() S: what does this do? J: I think this only matters if gym opens another window
 
    return V, errors, gameDurations, np.asarray(gamesWon)




### Analysis ###
# Error
def plotFromDict(errorDict, title = ""): 
    plt.clf() # Clears current figure
    plt.rcParams.update({'font.size': 12})
    
    # Errors per state or state,action pair: (inspired by https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b)
    errorLists = [list(x)[:nrEpisodes] for x in errorDict.values()]
    for errorList in errorLists:
        plt.plot(errorList)
    
    plt.xlabel('Number of Visits')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()
    plt.savefig("FrozenLakeError-"+title+".pdf", bbox_inches = 'tight')
    
# Learning curve
def plotLearningCurve(gamesWon, title): 
    plt.clf()
    plt.rcParams.update({'font.size': 12})
    
    x = np.arange(1,nrEpisodes+1)
    y = np.cumsum(gamesWon, dtype=float)/x
    plt.plot(x,y)
    
    plt.xlabel('Episode')
    plt.ylabel('Average reward per episode')
    plt.title('Learning cursve '+title)
    plt.show()
    plt.savefig("FrozenLakeLC-"+title+".pdf", bbox_inches = 'tight')


# Quality check of final policy
def averagePerformance(Q, policy = epsGreedy): 
    '''Performs same policy over and over to measure accuracy
    
    Especially meant to test final policy developed during simulation'''
    totalRewards = 0
    nrGames = 1000
    
    for i in range(nrGames):
        done = False
        state = env.reset()
        
        while not done: 
            action = policy(Q, state)
            state, reward, done, info = env.step(action)
            totalRewards += reward
    
    return totalRewards/nrGames #TODO sth with confidence intervals (also for other analyses)


# Summary
def printGameSummary(durations, evaluationMethod):
    print(evaluationMethod,":")
    print(f"Percentage of games won: {(len(durations)/nrEpisodes)*100}")
    print(f"Average duration winning game: {np.mean(durations)} steps")



### Execution ###
createEnv()

nrEpisodes = 50000
alpha = 0.02 # stepsize
gamma = 1 # discounting rate; may differ per state and iteration
eps = 0.05

def runSimulation(evaluationMethod):
    global Q, durations, gamesWon
    Q, errors, durations, gamesWon = policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod = "SARSA", epsilon = eps, printSteps = False)
    plotFromDict(errors, evaluationMethod)
    plotLearningCurve(gamesWon, evaluationMethod)
    printGameSummary(durations, evaluationMethod)


# # TD: 
# evaluationMethod = "TD"
# runSimulation(evaluationMethod)
# print(averagePerformance(Q))

# # Q-learning:
# evaluationMethod = "Q"
# runSimulation(evaluationMethod)
# print(averagePerformance(Q))

# SARSA:
evaluationMethod = "SARSA"
runSimulation(evaluationMethod)
print(averagePerformance(Q))




