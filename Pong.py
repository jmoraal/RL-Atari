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

### Initialise game settings etc.
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


def summaryStats(data, confidence = 0.95): 
    mean = np.mean(data)
    std = np.std(data)
    confInt = st.t.interval(0.95, len(data)-1, loc=mean, scale=st.sem(data)) 
    
    return mean, std, confInt

def reduceState(state): 
    ''' Reduces dimension of state. '''
    state = state[34:194,:,:] # remove top 34 and bottom 16 pixels
    state = state[:,:,0] # remove third axis (color channels)
    state = np.where(state == 109, 0, state) # brown is background so set those values to 0
    state = np.where(state != 0, 255, state) # set all other values (green, white, orange) to 255
    
    state = resize(state, (5, 5)) # downscale image
    state = np.where(state != 0, 1, 0)
    state = state.flatten()
    state = state.dot(2**np.arange(state.size)[::-1]) # convert binary array to integer (=number of state) using dot product

    return state
    
# Main function: policy evaluation/improvement                    
def policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod , epsilon = 0, printSteps = True): 
    global V, gameDurations, endScore
    
    gameDurations = [] # for analysis
    endScore = np.zeros(nrEpisodes)    
    # winRatios = np.zeros(progressPoints)
    # confIntervals = np.zeros((progressPoints,2))
    # valueUpdates = np.zeros([nrStates, progressPoints+1])
    
    # Initialize value function and error lists
    if evaluationMethod == "TD":
        V = np.zeros(nrStates) 
    elif evaluationMethod == "Q":
        V = np.ones((nrStates,nrActions)) 
    elif evaluationMethod == "SARSA":
        V = np.ones((nrStates,nrActions))
        action = env.action_space.sample() 
    elif evaluationMethod == "DoubleQ": 
        Vdouble = np.ones((nrStates,nrActions,2)) #instead of Q1 and Q2, initializa one array with extra axis
        errors = {i:list() for i in range(nrStates*nrActions)}

        
    # Run game nrEpisodes times
    for n in range(nrEpisodes):
        currentState = env.reset() # reset game to initial state
        currentState = reduceState(currentState)
        
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
                # print(action)
                
                # Take chosen action, visit new state and obtain reward
                newState, reward, done, info = env.step(action)
                newState  = reduceState(newState)
                
                # Update value function
                tempValue = V[currentState]
                V[currentState] += alpha*(reward + gamma*V[newState] - tempValue) 
                # errors[currentState].append(float(np.abs(tempValue-V[currentState])))
            
            #Q-learning:
            elif evaluationMethod == "Q":
                action = epsGreedy(V, currentState, epsilon = epsilon)
                
                newState, reward, done, info = env.step(action)
                newState  = reduceState(newState)
                
                tempValue = V[currentState, action]
                tempMax = np.max(V[newState,:])
                V[currentState,action] += alpha*(reward + gamma*tempMax - tempValue)
                    
                # errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) # TODO or different error?
                #TODO Also, error now changes with alpha, book does not do this; what should we take?
                
            #SARSA:
            elif evaluationMethod == "SARSA":
                # action = epsGreedy(V, currentState, epsilon = epsilon)
                newState, reward, done, info = env.step(action)
                newState  = reduceState(newState)
                newAction = epsGreedy(V, newState, epsilon = epsilon)
                
                tempValue = V[currentState, action]
                V[currentState,action] += alpha*(reward + gamma*V[newState,newAction] - V[currentState, action])
                # errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) # TODO or different error?
                action = newAction 
            
            #Double-Q learning: (avoids maximisation bias)
            elif evaluationMethod == "DoubleQ": 
                #Pick action epsilon-greedy from Q1+Q2 (via axis-sum)
                action = epsGreedy(np.sum(Vdouble, axis = 2), currentState, epsilon = epsilon)
                
                newState, reward, done, info = env.step(action)
                newState  = reduceState(newState)
                
                ind = np.random.randint(2) #chooses which array to update
                
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
        
        print(f"Episode {n} finished after {t+1} timesteps with reward {reward}")
        gameDurations.append(t+1) # won the game
        endScore[n] = reward
        
        epsilon *= decay_rate
        epsilon = max(epsilon, min_epsilon)
        
        # interval = nrEpisodes//progressPoints #recall that '//' is division without remainder
        # if ((n+1) % interval == 0): #Print progress given number of times (evenly distributed)
        #     epsilon *= decay_rate
        #     epsilon = max(epsilon, min_epsilon)
    
        #     if evaluationMethod == "DoubleQ": 
        #         V = np.sum(Vdouble,axis = 2)
                
        #     if evaluationMethod == "TD":
        #         valueUpdates[:,n//interval] = V 
        #         print(f"{n+1} out of {nrEpisodes}")
        #     else:
        #     #if (evaluationMethod == "SARSA" or evaluationMethod == "Q"): 
        #         ratio, _, confInt = policyPerformanceStats(V) # Note: significantly slows down iteration as 
        #                                                 # this function iterates the game a few hundred times. 
        #                                                 # Still, deemed an important progress measure. 
        #                                                 # TODO? Maybe can be replaced by sliding-window type 
        #                                                 # average (instead of running average as we use now in evaluation)
        #         winRatios[n//interval] = ratio
        #         confIntervals[n//interval,:] = confInt
        #         print(f"{n+1} out of {nrEpisodes}, current win ratio is {ratio:3.4f}. eps {epsilon}")
         
    
    return V, gameDurations, np.asarray(endScore)


### Analysis ###
def plotInitialize():
    plt.clf() # Clears current figure
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 12})

# Summary
def printGameSummary(durations, endScores, evaluationMethod):
    print(evaluationMethod,":")
    print(f"Mean end score: {np.mean(endScores)}")
    print(f"Mean game duration: {np.mean(durations)}")
    print(f"Mean end score in last 100 games: {np.mean(endScores[-100:len(endScores)])}")



### Execution ###
createEnv() # create game 
env.close()

nrEpisodes =2
alpha = .02 # stepsize
gamma = 1 # discounting rate; may differ per state and iteration
eps = .9 # initial value
decay_rate = .5
min_epsilon = 0.05

def runSimulation(evaluationMethod):
    global V, valueUpdates, errors, durations, gamesWon, winRatios, confIntervals
    V, durations, endScores = policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod = evaluationMethod, epsilon = eps, printSteps = False)
                                                                                               
    # plotFromDict(errors, evaluationMethod)
    printGameSummary(durations, endScores, evaluationMethod)
    # plotWinRatios(winRatios, confIntervals, evaluationMethod, len(gamesWon)//len(winRatios)) 
    
    #final policy evaluation:
    # if (evaluationMethod == "SARSA" or evaluationMethod == "Q"): 
        # win, std, CI = policyPerformanceStats(V, nrGames = 10000)
        # print("Final policy has winning ratio {:.3} with confidence interval [{:.3},{:.3}]".format(win, CI[0], CI[1]))
    

# TD: 
# evaluationMethod = "TD"

# Q-learning:
evaluationMethod = "Q"

# SARSA:
# evaluationMethod = "SARSA"

#Double Q-learning: 
# evaluationMethod = "DoubleQ"

runSimulation(evaluationMethod)

# for i in range(210):
#     for j in range(160):
        
#         color = env.reset()[i,j,0]
#         # print(color)
#     # if i > 33 and i < 194:
#         if  color != 109 and color != 53 and color != 0:
#             print(env.reset()[i,j])