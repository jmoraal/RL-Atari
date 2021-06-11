# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:05:05 2021

@author: s161981, s1017389
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
    nrActions  = env.nA
    startState = 0
    finalState = nrStates - 1
    returns = np.zeros(nrStates)


def epsGreedy(Q, state, epsilon = 0.05): 
    ''' Makes epsilon-greedy choice for action given state and value table'''
    if np.random.rand() > epsilon: # with probability 1-epsilon, choose current best option greedily
        return np.argmax(Q[state,:])
    else: # with probability epsilon, choose randomly
        return env.action_space.sample()


# Quality check of given policy
# def averagePerformance(Q, policy = epsGreedy): 
#     '''Performs same policy over and over to measure accuracy
    
#     Especially meant to test final policy developed during simulation'''
#     nrGames = 1000
#     rewards = np.zeros(nrGames)
    
#     for i in range(nrGames):
#         done = False
#         state = env.reset()
        
#         while not done: 
#             action = policy(Q, state)
#             state, reward, done, info = env.step(action)
        
#         rewards[i] = reward
    
#     return np.mean(rewards), np.std(rewards) #TODO sth with confidence intervals (also for other analyses)

## Main function: policy evaluation/improvement                    
def policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod , epsilon = 0, printSteps = True): 
    
    gameDurations = [] # for analysis
    gamesWon = np.zeros(nrEpisodes)
    winRatios = np.zeros(progressPoints)
    valueUpdates = np.zeros([nrStates, progressPoints+1])
    counter = 1
    
    # Initialize value function and error lists
    if evaluationMethod == "TD":
        V = np.zeros(nrStates) 
        errors = {i:list() for i in range(nrStates)}  # keeps track of error in each state
        valueUpdates[:,0] = V
    elif evaluationMethod == "Q":
        # V = np.random.rand(nrStates,nrActions)
        V = np.ones((nrStates,nrActions))
        errors = {i:list() for i in range(nrStates*nrActions)}  # 2d matrix mapped to vector! J: Why? Doesn't it only make indexing difficult further on
    elif evaluationMethod == "SARSA":
        # V = np.random.rand(nrStates,nrActions)
        V = np.ones((nrStates,nrActions))
        
        #V = np.zeros((nrStates,nrActions))
        #V[finalState,:] = 1
        
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
                action = env.action_space.sample()
                
                # Take chosen action, visit new state and obtain reward
                newState, reward, done, info = env.step(action)
                
                # Update value function
                tempValue = V[currentState]
                V[currentState] += alpha*(reward + gamma*V[newState] - tempValue) 
                errors[currentState].append(float(np.abs(tempValue-V[currentState])))
            
            #Q-learning:
            elif evaluationMethod == "Q":
                action = epsGreedy(V, currentState, epsilon = epsilon)
                
                newState, reward, done, info = env.step(action)
                
                tempValue = V[currentState, action]
                tempMax = np.max(V[newState,:])
                V[currentState,action] += alpha*(reward + gamma*tempMax - tempValue)
                    
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) #TODO S: same errors as TD ??
                #J: In this formulation yes; but errors are not actually equal as they depend on the update. 
                #TODO Also, error now changes with alpha, book does not do this; what should we take?
                
            #SARSA:
            elif evaluationMethod == "SARSA":
                action = epsGreedy(V, currentState, epsilon = epsilon)
                newState, reward, done, info = env.step(action)
                newAction = epsGreedy(V, newState, epsilon = epsilon)
                
                tempValue = V[currentState, action]
                V[currentState,action] += alpha*(reward + gamma*V[newState,newAction] - tempValue)
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
        gamesWon[n] = reward
        
        
        interval = nrEpisodes//progressPoints
        if ((n+1) % interval == 0): #Print progress given number of times (evenly distributed)
            print(f"{n+1} out of {nrEpisodes}")
            
            if evaluationMethod == "TD":
                valueUpdates[:,counter] = V #TODO kan wss netter dan met die counter
                counter += 1
            # if (evaluationMethod == "SARSA" or evaluationMethod == "Q"): 
            #     _, ratio = averagePerformance(V) # Note: significantly slows down iteration as this function iterates the game a few hundred times
            #                                      # Still, deemed an important progress measure
            #                                      #TODO? Maybe can be replaced by sliding-window type average (instead of running average as we use now in evaluation)
            #     winRatios[n//interval] = ratio
            #     print(f"{n+1} out of {nrEpisodes}, current win ratio is {ratio:3.4f}")
            # else: 
                # print(f"{n+1} out of {nrEpisodes}")
        
            
            # Update policy using value function
            # Now that we have the value function of all the states, our next step is to extract the policy from the Value Function.
            # See https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
            # policy = updatePolicy(...)
            
            #J: is this still relevant? At least for Q and SARSA, policy is defined by Q (V) and hence implicitly updated
         
    return V, valueUpdates, errors, gameDurations, np.asarray(gamesWon), winRatios




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
    plt.title('Error')
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
    plt.title('Learning curve '+title)
    plt.show()
    plt.savefig("FrozenLakeLC-"+title+".pdf", bbox_inches = 'tight')


# Winning ratio development over time
def plotWinRatios(winRatios, title, interval): 
    plt.clf()
    plt.rcParams.update({'font.size': 12})
    
    x = np.arange(nrEpisodes//interval)*interval
    y = winRatios
    plt.plot(x,y)
    
    plt.xlabel('Episode')
    plt.ylabel('Average reward per episode')
    plt.title('Learning curve '+title)
    plt.show()
    plt.savefig("FrozenLakeLC-"+title+".pdf", bbox_inches = 'tight')

def plotValueUpdates(valueUpdates, trueV): # exclude final state 15 since this is not interesting 
    plt.clf()
    plt.rcParams.update({'font.size': 12})
    
    for i in range(4):
        plt.plot(valueUpdates[0:nrStates-1,i], label = i*nrEpisodes//progressPoints)
    plt.plot(trueV[0:nrStates-1], label = "True")
    # plt.plot(valueUpdates)
    # plt.plot(trueV)
    
    plt.xlabel('State')
    plt.ylabel('')
    plt.title('Value updates')
    plt.show()
    plt.legend()#prop={'size': 16})
    plt.savefig("FrozenLakeVU.pdf", bbox_inches = 'tight')

# Summary
def printGameSummary(durations, gamesWon, evaluationMethod, winRatios):
    print(evaluationMethod,":")
    print(f"Percentage of games won: {(len(durations)/nrEpisodes)*100}")
    last = gamesWon[-nrEpisodes//10:]
    print(f"Percentage of games won towards end: {np.sum(last)/len(last)*100}")
    print(f"Average duration winning game: {np.mean(durations)} steps") #wild idea: can we add a penalty to all non-Goal states so that the policy will favour a shorter solution? :D
    if (evaluationMethod == "SARSA" or evaluationMethod == "Q"): #TODO not yet working for TD
        print(f"Final winning ratio: {winRatios[-1]}")  



### Execution ###
createEnv()

nrEpisodes = 1000000
alpha = .001 # stepsize
gamma = 1 # discounting rate; may differ per state and iteration
eps = .05
progressPoints = 4

def runSimulation(evaluationMethod):
    global V, valueUpdates, errors, durations, gamesWon, winRatios, progressPoints
    V, valueUpdates, errors, durations, gamesWon, winRatios = policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod = evaluationMethod, epsilon = eps, printSteps = False)
    # plotFromDict(errors, evaluationMethod)
    # plotLearningCurve(gamesWon, evaluationMethod)
    printGameSummary(durations, gamesWon, evaluationMethod, winRatios)
    plotWinRatios(winRatios, evaluationMethod, len(gamesWon)//len(winRatios)) 
    if evaluationMethod == "TD":
        plotValueUpdates(valueUpdates, trueV)
    #TODO somehow, winning ratio does not seem to get above exactly 0.5 (for SARSA at least). Could mean ratio computation
    # is wrong, or update rule somehow stops improving (thinks its objective is reached) after that...


# # TD: 
# evaluationMethod = "TD"
# runSimulation(evaluationMethod)
# True value function (computed with Mathematica)
trueV = np.array([0.0139398, 0.0116309, 0.020953, 0.0104765, 0.0162487, 0, 0.0407515, 0, 0.0348062, 0.0881699, 0.142053, 0, 0, 0.17582, 0.439291, 1])
# V[15]=1
# print(V)
# print("overall error:")
# print(np.sum(np.abs(V-trueV)))

# Q-learning:
evaluationMethod = "Q"
runSimulation(evaluationMethod)
# plotFromDict(errors, evaluationMethod)
plotLearningCurve(gamesWon, evaluationMethod)
printGameSummary(durations, gamesWon, evaluationMethod, winRatios)
plotWinRatios(winRatios, evaluationMethod, len(gamesWon)//len(winRatios)) 

# # SARSA:
# evaluationMethod = "SARSA"
# runSimulation(evaluationMethod)

# for i in range(0,16):
#     for j in range(0,4):
#         print(i,j)
#         print(env.P[i][j])
        


# plotValueUpdates(valueUpdates, trueV)


