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

### Initialize game settings etc.
def createEnv(size = 4): 
    ''' Create game environment from Gym package. '''
    global nrStates, nrActions, env
    
    if size == 8:
        env = gym.make ("FrozenLake8x8-v0")
    else:
        env = gym.make ("FrozenLake-v0")
        
    nrStates = env.nS
    nrActions  = env.nA

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

def summaryStats(data, confidence = 0.95): 
    ''' Given an array, returns mean, standard deviation and 95% confidence interval '''
    mean = np.mean(data)
    std = np.std(data)
    confInt = st.t.interval(0.95, len(data)-1, loc=mean, scale=st.sem(data)) 
    
    return mean, std, confInt
    
# Quality check of given policy
def policyPerformanceStats(Q, policy = greedy, nrGames = 1000): 
    '''Performs same given policy over and over to measure accuracy, 
    outputs mean, std and confidence interval of mean
    
    1000 games is enough for an indication of progress; a lot more would 
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
    
    if evaluationMethod == "TD":
        progressPoints = 4
    else: 
        progressPoints = 100
    
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
        errors = {i:list() for i in range(nrStates*nrActions)}  
    elif evaluationMethod == "SARSA":
        V = np.ones((nrStates,nrActions))
        errors = {i:list() for i in range(nrStates*nrActions)} 
        action = env.action_space.sample()  # needs to be initialized for SARSA
    elif evaluationMethod == "DoubleQ": 
        Vdouble = np.ones((nrStates,nrActions,2)) #instead of Q1 and Q2, initialize one array with extra axis
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
                print("ok")
            # Evaluate policy
            #Temporal difference:
            if evaluationMethod == "TD":
                action = env.action_space.sample()
                #print(action)
                
                # Take chosen action, visit new state and obtain reward
                newState, reward, done, info = env.step(action)
                
                # Update value function
                tempValue = V[currentState]
                V[currentState] += alpha*(reward + gamma*V[newState] - tempValue) 
                errors[currentState].append(float(np.abs(reward + gamma*V[newState] - tempValue)))
            
            #Q-learning:
            elif evaluationMethod == "Q":
                action = epsGreedy(V, currentState, epsilon = epsilon)
                
                newState, reward, done, info = env.step(action)
                tempValue = V[currentState, action]
                tempMax = np.max(V[newState,:])
                V[currentState,action] += alpha*(reward + gamma*tempMax - tempValue)
                    
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) #
                
            #SARSA:
            elif evaluationMethod == "SARSA":
                newState, reward, done, info = env.step(action)
                newAction = epsGreedy(V, newState, epsilon = epsilon)
                
                tempValue = V[currentState, action]
                V[currentState,action] += alpha*(reward + gamma*V[newState,newAction] - V[currentState, action])
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) 
                action = newAction 
            
            #Double-Q learning: (avoids maximisation bias)
            elif evaluationMethod == "DoubleQ": 
                ind = np.random.randint(2) #chooses which array to update
                action = epsGreedy(Vdouble[:,:,ind], currentState, epsilon = epsilon)
                #Pick action epsilon-greedy from Q1+Q2 (via axis-sum)
                # action = epsGreedy(np.sum(Vdouble, axis = 2), currentState, epsilon = epsilon)
                
                newState, reward, done, info = env.step(action)
                
                
                tempValue = Vdouble[currentState, action, ind]
                maxAction = greedy(Vdouble[:,:, ind],newState)#np.argmax(Vdouble[newState,:, ind])
                Vdouble[currentState,action, ind] += alpha*(reward + gamma*Vdouble[newState, maxAction, 1-ind] - tempValue)
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-Vdouble[currentState, action,ind]))) 
            
            # Update state
            currentState = newState
            t += 1
        
        #if evaluationMethod == "Janne": 
            '''idea: keep track of path, remove cycles and do not go to last 
            state before terminal unless reward is 1.   
            can we add a penalty to all non-Goal states so that the policy will favour a shorter solution? 
            '''
        
        # Print results if desired
        if printSteps:
            directions =  ["L", "D", "R", "U"]
            print("At time", t, ", we obtain reward", reward, ", choose ", directions[action], " and move to:", newState, "\n")
        
        if printSteps: print(f"Episode finished after {t+1} timesteps" )
        if reward == 1: gameDurations.append(t+1) # won the game
        gamesWon[n] = reward
        
        
        interval = nrEpisodes//progressPoints #recall that '//' is division without remainder
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
                #ideally, the following line would work and save running time
                # ratio, _, confInt = summaryStats(gamesWon[-10:]) 
                #instead, the following:
                ratio, _, confInt = policyPerformanceStats(V) 
                # Note: significantly slows down iteration as 
                # this function iterates the game a few hundred times. 
                # Still, deemed an important progress measure.
                
                winRatios[n//interval] = ratio
                confIntervals[n//interval,:] = confInt
                print(f"{n+1} out of {nrEpisodes}, current win ratio is {ratio:3.4f}. eps {epsilon}")
         
    
    return V, valueUpdates, errors, gameDurations, np.asarray(gamesWon), winRatios, confIntervals


### Analysis ###
def plotInitialize():
    '''Initialises plot settings (no args or output)'''
    plt.clf() # Clears current figure
    plt.figure(figsize=(10,5)) #set plot size
    plt.rcParams.update({'font.size': 12}) # set font size
    
def plotFromDict(errorDict, title = "", nrPoints = 100): 
    ''' Given a dictionary of arrays, plots first 100 entries of each'''
    plotInitialize()
    
    # Errors per state or state,action pair:
    errorLists = [list(x)[:nrEpisodes] for x in errorDict.values()]
    for errorList in errorLists: #extract arrays from dictionary
        plt.plot(errorList[:nrPoints]) #plot extracted array
    
    # Mark-up:
    plt.xlabel('Number of Visits')
    plt.ylabel('Error')
    plt.title('Update error per visit for each state')
    plt.savefig("FrozenLakeError-"+title+".pdf", bbox_inches = 'tight')
    plt.show()

def movingAverage(data, size):
    '''Computes average of data over sliding window of width [size], 
    returning array of length (len(data) - size)'''
    data = np.array(data)
    moment1 = np.convolve(data, np.ones(size), 'valid')/size #compute mean with sliding window
    moment2 = np.convolve(data**2, np.ones(size), 'valid')/size #compute 2nd moment with sliding window
    stds = np.sqrt(moment2 - moment1**2) #compute standard deviation elementwise from 1st and 2nd moment
    confInts = np.array(st.t.interval(0.95, size-1, 
                                      loc=moment1, 
                                      scale=stds/(np.sqrt(size - 1)))).transpose()
    
    return moment1, stds, confInts

# Winning ratio development over time
def plotWinRatios(winRatios, confIntervals, title, interval): 
    '''Plots winning ratios with given confidence intervals
    
    INPUT: 
        Array of winning ratios, length n
        Array of confidence intervals, shape (n,2) 
        Plot title (string)
        Interval (int), number of episodes per datapoint (to correctly label the axes)
    
    OUTPUT:
        none, only save to pdf
    '''
    plotInitialize()
    
    #winratios:
    #x = np.arange(nrEpisodes//interval)*interval
    x = np.arange(len(winRatios))*interval
    y = winRatios
    plt.plot(x,y, label = 'Winning ratios', color = 'b')
    
    #confidence intervals:
    plt.plot(x,confIntervals, 
             label = '95% Confidence intervals', 
             color = 'r', 
             linestyle = '--', 
             linewidth = 1)
    
    plt.xlabel('Episode')
    plt.ylabel('Average reward per episode')
    plt.title('Learning curve '+title)
    plt.savefig("FrozenLakeWR-"+title+".pdf", bbox_inches = 'tight')
    plt.show()

# Value function for TD learning
def plotValueUpdates(valueUpdates, trueV): # exclude final state 15 since this is not interesting 
    '''Given arrays of value-updates and true V, plots and saves both'''
    plotInitialize()
    
    for i in range(progressPoints):
        plt.plot(valueUpdates[0:nrStates-1,i], 
                 label = i*nrEpisodes//progressPoints, 
                 marker = 'o', 
                 markeredgewidth = .5)
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
    '''Prints summary of information on developed policy, games won, etc. '''
    print(evaluationMethod,":")
    print(f"Percentage of games won: {(len(durations)/nrEpisodes)*100}")
    last = gamesWon[-nrEpisodes//10:] #array of  last 10% of games
    print(f"Percentage of games won towards end: {np.sum(last)/len(last)*100}")
    durMean, durstd, durCI = summaryStats(durations)
    print(f"Average duration winning game: {durMean:.3f} steps, with CI {durCI}") 
    if (evaluationMethod == "SARSA" or evaluationMethod == "Q"): 
        print(f"Final winning ratio: {winRatios[-1]}")  
    #final policy evaluation: 
    if (evaluationMethod == "SARSA" or evaluationMethod == "Q" or evaluationMethod == "DoubleQ"):
        win, std, CI = policyPerformanceStats(V, nrGames = 10000)
        print("Final policy has winning ratio {:.3}".format(win),
              "with confidence interval [{:.3},{:.3}]".format(CI[0], CI[1]))



### Execution ###

def runSimulation(evaluationMethod):
    global V, valueUpdates, errors, durations, gamesWon, winRatios, confIntervals
    V, valueUpdates, errors, durations, gamesWon, winRatios, confIntervals = policyEvaluation(nrEpisodes, alpha, gamma, 
                                                                                              evaluationMethod = evaluationMethod, 
                                                                                              epsilon = eps, 
                                                                                              printSteps = False)
                                                                                               
    plotFromDict(errors, evaluationMethod)
    printGameSummary(durations, gamesWon, evaluationMethod, winRatios)
    plotWinRatios(winRatios, confIntervals, evaluationMethod, len(gamesWon)//len(winRatios)) 
    
    if evaluationMethod == "TD":
        # True value function (computed with Mathematica)
        V[15]=1
        trueV = np.array([0.0139398, 0.0116309, 0.020953,   0.0104765, 
                          0.0162487, 0,         0.0407515,  0, 
                          0.0348062, 0.0881699, 0.142053,   0, 
                          0,         0.17582,   0.439291,   1])
        #plotValueUpdates(valueUpdates, trueV)
        print("overall error:")
        print(np.sum(np.abs(V-trueV)))


createEnv() # create game 

# Simulation (hyper)parameters: 
nrEpisodes = 50000
alpha = .02 # stepsize
gamma = 1 # discounting rate; may differ per state and iteration
eps = .9 # initial value
decay_rate = 0.8
min_epsilon = 0.01
progressPoints = 100 # choose 4 for TD, about 100 for others


# Choose evaluation method: (uncomment corresponding line, run 'runSimulation')
# TD: 
# evaluationMethod = "TD"

# Q-learning:
# evaluationMethod = "Q"

# SARSA:
evaluationMethod = "SARSA"

#Double Q-learning: 
# evaluationMethod = "DoubleQ"


runSimulation(evaluationMethod)

