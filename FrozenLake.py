# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:05:05 2021

@author: s161981, s1017389
"""
import numpy as np
import gym
from matplotlib import pyplot as plt 
import scipy.stats as st
np.random.seed(1964)

### Initialise game settings etc.
def createEnv(game, size = 4): 
    ''' Create game environment from Gym package.'''
    global nrStates, nrActions, returns, env
    
    if game == "Pong":
        env = gym.make ("Pong-v0")
        nrStates = 256**(210*160*3)
        nrActions = 5
    elif game == "FrozenLake":  
        if size == 8:
            env = gym.make ("FrozenLake8x8-v0")
        else:
            env = gym.make ("FrozenLake-v0")
        
        nrStates = env.nS
        nrActions  = env.nA
        returns = np.zeros(nrStates)

def greedy(Q, state):
    ''' Makes greedy choice for action given state and value table'''
    argmaxes = np.flatnonzero(Q[state,:] == np.max(Q[state,:]))
    return np.random.choice(argmaxes)

def epsGreedy(Q, state, epsilon = 0.05): 
    ''' Makes epsilon-greedy choice for action given state and value table'''
    if np.random.rand() > epsilon: # with probability 1-epsilon, choose current best option greedily
        return greedy(Q,state)
    else: # with probability epsilon, choose randomly
        return env.action_space.sample()


def summaryStats(data, confidence = 0.95): 
    mean = np.mean(data)
    std = np.std(data)
    confInt = st.t.interval(0.95, len(data)-1, loc=mean, scale=st.sem(data)) 
    
    return mean, std, confInt
    
# Quality check of given policy
def policyPerformanceStats(Q, policy = greedy, nrGames = 2000): 
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
        gameReward = 0
        
        while not done: 
            action = policy(Q, state)
            state, stepReward, done, info = env.step(action)
            gameReward += stepReward
        
        rewards[i] = gameReward
    
    return summaryStats(rewards)

# Main function: policy evaluation/improvement                    
def policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod , epsilon = 0, printSteps = True): 
    
    gameDurations = [] # for analysis
    gamesWon = np.zeros(nrEpisodes)
    winRatios = np.zeros(progressPoints)
    confIntervals = np.zeros((progressPoints,2))
    valueUpdates = np.zeros([nrStates, progressPoints+1])
    
    # Initialize value function and error lists
    if evaluationMethod == "TD":
        V = np.zeros(nrStates) 
        errors = {i:list() for i in range(nrStates)}  # keeps track of error in each state
        valueUpdates[:,0] = V
    elif evaluationMethod == "Q":
        V = np.ones((nrStates,nrActions))
        errors = {i:list() for i in range(nrStates*nrActions)}  # 2d matrix mapped to vector! J: Why? Doesn't it only make indexing difficult further on
    elif evaluationMethod == "SARSA":
        V = np.ones((nrStates,nrActions))
        errors = {i:list() for i in range(nrStates*nrActions)} 
        action = env.action_space.sample()  # needs to be initialized for SARSA
    elif evaluationMethod == "DoubleQ": 
        Vdouble = np.ones((nrStates,nrActions,2)) #instead of Q1 and Q2, initialise one array with extra axis
        errors = {i:list() for i in range(nrStates*nrActions)}

        
    # Run game nrEpisodes times
    for n in range(nrEpisodes):
        currentState = env.reset() # reset game to initial state
        done = False
        t = 0
    
        
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
                    
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) # TODO or different error?
                #TODO Also, error now changes with alpha, book does not do this; what should we take?
                
            #SARSA:
            elif evaluationMethod == "SARSA":
                # action = epsGreedy(V, currentState, epsilon = epsilon)
                newState, reward, done, info = env.step(action)
                newAction = epsGreedy(V, newState, epsilon = epsilon)
                
                tempValue = V[currentState, action]
                V[currentState,action] += alpha*(reward + gamma*V[newState,newAction] - V[currentState, action])
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) # TODO or different error?
                action = newAction 
            
            #Double-Q learning: (avoids maximisation bias)
            elif evaluationMethod == "DoubleQ": 
                #Pick action epsilon-greedy from Q1+Q2 (via axis-sum)
                action = epsGreedy(np.sum(Vdouble, axis = 2), currentState, epsilon = epsilon)
                
                newState, reward, done, info = env.step(action)
                
                ind = np.random.randint(2) #chooses which array to update
                
                tempValue = Vdouble[currentState, action, ind]
                maxAction = greedy(Vdouble[:,:, ind],newState)#np.argmax(Vdouble[newState,:, ind])
                Vdouble[currentState,action, ind] += alpha*(reward + gamma*Vdouble[newState, maxAction, 1-ind] - tempValue)
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-Vdouble[currentState, action,ind]))) 
                
            
            
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
            epsilon *= decay_rate
            epsilon = max(epsilon, min_epsilon)
    
            if evaluationMethod == "DoubleQ": 
                V = np.sum(Vdouble,axis = 2)
                
            if evaluationMethod == "TD":
                valueUpdates[:,n//interval] = V 
                print(f"{n+1} out of {nrEpisodes}")
            else:
            #if (evaluationMethod == "SARSA" or evaluationMethod == "Q"): 
                ratio, _, confInt = policyPerformanceStats(V) # Note: significantly slows down iteration as 
                                                        # this function iterates the game a few hundred times. 
                                                        # Still, deemed an important progress measure. 
                                                        # TODO? Maybe can be replaced by sliding-window type 
                                                        # average (instead of running average as we use now in evaluation)
                winRatios[n//interval] = ratio
                confIntervals[n//interval,:] = confInt
                print(f"{n+1} out of {nrEpisodes}, current win ratio is {ratio:3.4f}. eps {epsilon}")
         
    
    return V, valueUpdates, errors, gameDurations, np.asarray(gamesWon), winRatios, confIntervals


### Analysis ###
def plotInitialize():
    plt.clf() # Clears current figure
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 12})
    
def plotFromDict(errorDict, title = ""): 
    plotInitialize()
    
    # Errors per state or state,action pair: (inspired by https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b)
    errorLists = [list(x)[:nrEpisodes] for x in errorDict.values()]
    for errorList in errorLists:
        plt.plot(errorList)
    
    plt.xlabel('Number of Visits')
    plt.ylabel('Error')
    plt.title('Error')
    plt.savefig("FrozenLakeError-"+title+".pdf", bbox_inches = 'tight')
    plt.show()

# Winning ratio development over time
def plotWinRatios(winRatios, confIntervals, title, interval): 
    # more informative method than learning curve!
    plotInitialize()
    
    #winratios:
    x = np.arange(nrEpisodes//interval)*interval
    y = winRatios
    plt.plot(x,y, label = 'Winning ratios', color = 'b')
    
    #confidence intervals:
    plt.plot(x,confIntervals, label = '95% Confidence intervals', color = 'r', linestyle = '--')
    
    plt.xlabel('Episode')
    plt.ylabel('Average reward per episode')
    plt.title('Learning curve '+title)
    plt.savefig("FrozenLakeWR-"+title+".pdf", bbox_inches = 'tight')
    plt.show()

# Value function for TD learning
def plotValueUpdates(valueUpdates, trueV): # exclude final state 15 since this is not interesting 
    plotInitialize()
    
    for i in range(progressPoints):
        plt.plot(valueUpdates[0:nrStates-1,i], label = i*nrEpisodes//progressPoints, marker = 'o', markeredgewidth = .5)
    plt.plot(trueV[0:nrStates-1], label = "True", marker = 's', markeredgewidth = 1)
    # plt.plot(valueUpdates)
    # plt.plot(trueV)
    
    plt.xlabel('State')
    plt.ylabel('')
    plt.title('Value updates')
    plt.legend()#prop={'size': 16})
    plt.savefig("FrozenLakeVU.pdf", bbox_inches = 'tight')
    # plt.show()

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
game = "Pong"
# game = "FrozenLake"
createEnv(game) # create game 

nrEpisodes = 100000
alpha = .02 # stepsize
gamma = 1 # discounting rate; may differ per state and iteration
eps = .07 # initial value
decay_rate = 1
min_epsilon = 0.07
progressPoints = 100 # choose 4 for TD, about 100 for others

def runSimulation(evaluationMethod):
    global V, valueUpdates, errors, durations, gamesWon, winRatios, confIntervals
    V, valueUpdates, errors, durations, gamesWon, winRatios, confIntervals = policyEvaluation(nrEpisodes, alpha, gamma, 
                                                                                              evaluationMethod = evaluationMethod, 
                                                                                              epsilon = eps, 
                                                                                              printSteps = False)
                                                                                               
    plotFromDict(errors, evaluationMethod)
    printGameSummary(durations, gamesWon, evaluationMethod, winRatios)
    plotWinRatios(winRatios, confIntervals, evaluationMethod, len(gamesWon)//len(winRatios)) 
    print(gamesWon)
    
    #final policy evaluation:
    if (evaluationMethod == "SARSA" or evaluationMethod == "Q"): 
        win, std, CI = policyPerformanceStats(V, nrGames = 10000)
        print("Final policy has winning ratio {:.3} with confidence interval [{:.3},{:.3}]".format(win, CI[0], CI[1]))
    
    if evaluationMethod == "TD":
        # True value function (computed with Mathematica)
        V[15]=1
        trueV = np.array([0.0139398, 0.0116309, 0.020953,   0.0104765, 
                          0.0162487, 0,         0.0407515,  0, 
                          0.0348062, 0.0881699, 0.142053,   0, 
                          0,         0.17582,   0.439291,   1])
        plotValueUpdates(valueUpdates, trueV)
        print("overall error:")
        print(np.sum(np.abs(V-trueV)))


# TD: 
# evaluationMethod = "TD"

# Q-learning:
# evaluationMethod = "Q"

# SARSA:
# evaluationMethod = "SARSA"

#Double Q-learning: 
evaluationMethod = "DoubleQ"

runSimulation(evaluationMethod)



