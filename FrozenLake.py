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
                    
def policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod, epsilon = 0, printSteps = True): 
    
    gameDurations = [] # for analysis
    gamesWon = []
    
    # Initialize value function and error lists
    if evaluationMethod == "TD":
        V = np.random.rand(nrStates) 
        V[finalState] = 0
        errors = {i:list() for i in range(nrStates)}  # keeps track of error in each state
    elif evaluationMethod == "Q":
        V = np.random.rand(nrStates,nrActions)
        V[finalState,:] = 0
        errors = {i:list() for i in range(nrStates*nrActions)}  # 2d matrix mapped to vector!
    elif evaluationMethod == "SARSA":
        V = np.random.rand(nrStates,nrActions)
        V[finalState,:] = 0
        errors = {i:list() for i in range(nrStates*nrActions)} 
        currentState = startState
        action = env.action_space.sample() #TODO # needs to be initialized for SARSA
        
        
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
            if evaluationMethod == "TD":
                action = env.action_space.sample()
                
                # Take chosen action, visit new state and obtain reward
                newState, reward, done, info = env.step(action)
                
                # Update value function
                tempValue = V[currentState]
                V[currentState] += alpha*(reward + gamma*V[newState] - V[currentState]) # S: or should we work with a np copy of V (like in tutorial)?
                errors[currentState].append(float(np.abs(tempValue-V[currentState])))
            elif evaluationMethod == "Q":
                if np.random.rand() > epsilon: # with probability 1-epsilon, choose current best option
                    action = np.argmax(V[currentState,:])
                else: # with probability epsilon, choose greedily
                    action = np.random.randint(4)
                
                newState, reward, done, info = env.step(action)
            
                tempValue = V[currentState, action]
                a_max = np.argmax(V[newState,:]) # find optimal new action
                V[currentState,:] += alpha*(reward + gamma*V[newState,a_max] - V[currentState, action])
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) # S: same errors as TD ??
                
            elif evaluationMethod == "SARSA":
                if np.random.rand() > epsilon: action = np.argmax(V[currentState,:])
                else: action = np.random.randint(4)
                newState, reward, done, info = env.step(action)
                if np.random.rand() > epsilon: newAction = np.argmax(V[newState,:])
                else: newAction = np.random.randint(4)
                tempValue = V[currentState, action]
                V[currentState,action] += alpha*(reward + gamma*V[newState,newAction] - V[currentState, action])
                errors[currentState*nrActions + action].append(float(np.abs(tempValue-V[currentState, action]))) # S: same errors as TD ??
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
            
            
               
        
#         env.close() S: what does this do?
    
        
        
    

    return V, errors, gameDurations, np.asarray(gamesWon)


# def TD(nrEpisodes, alpha, gamma, policy = None, printSteps=True): 
#     ''' Runs TD(0) algorithm for given number of episodes to estimate value function V '''
    
#     V = np.zeros(nrStates) #TODO probably initialise this inside TD, not here
#     V[finalState] = 1
#     errPerState = {i:list() for i in range(nrStates)} # keeps track of error in each state.
#     # (note that tracking all errors in one list does not make sense, since some states
#     #  are visited more often earlier on and thus already show some convergence.)
#     for n in range(nrEpisodes):
#         env.reset() # Reset game to initial state
#         state = 0
#         t = 0
#         # Run the game
#         while True:
#             if printSteps: env.render()
            
#             # Choose action 
#             if policy == None:
#                 action = env.action_space.sample() #choose random action
#             else: 
#                 action = policy[state] #follow policy
            
#             # Take chosen action, visit new state and obtain reward
#             newState, reward, done, info = env.step(action)
            
#             # Update V and keep track of errors:
#             err = reward + gamma * V[newState] - V[state]
#             V[state] += alpha * err
#             errPerState[state].append(err) # TODO: now all negative... but at least converging
#             #not too happy with appending, but cannot know in advance how long it will become



#             #also, is the indexing correct? Book mentions this error is not available until next timestep (below eq 6.5)



                # S: We are working with 2 time steps (t+1 and t) or do you mean something else?
            
            
#             #TODO: sometimes, done like this: see e.g. https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b
#             # old = V[state] # must be stored here, for the case state = newState
#             # V[newState] += alpha * (reward + gamma * V[final] - V[newState])
            
#             # # Keep track of errors: now all negative... but at least converging
#             # TDerrPerState[state].append(reward + gamma*V[newState] - old)
                
#             state = newState
#             t += 1
            
#             if printSteps: 
#                 print("At time", t, ", we obtained reward", reward, ", and visited:", newState, "\n")
#                 # print("Next action:", actions[action])
            
#             if done:
#                 if printSteps: print(f"Episode finished after {t+1} timesteps" )
#                 break
        
#         env.close()
#     return V, errPerState


# ### Q-learning: ###
# #after Sutton & Barto, p131

# def Qlearning(nrEpisodes, alpha, gamma, epsilon, printSteps=True):
#     errPerState = {i:list() for i in range(nrStates)} # keeps track of error in each state.
#     for n in range(nrEpisodes):
#         env.reset() # Reset environment every episode?
#         state = 0
#         t = 0
        
#         # Run the game
#         while True:
#             if printSteps: env.render()
            
#             #Choose action using eps-greedy policy from Q
#             if np.random.rand() >= epsilon: # with probability 1-epsilon, choose greedily
#                 action = np.argmax(Q[state,:]) 
#             else: # with probability epsilon, do not choose greedy
#                 action = env.action_space.sample() #chooses random action
            
#             # Take chosen action, visit new state and obtain reward
#             newState, reward, done, info = env.step(action)
            
#             # Update Q and save error to state:
#             err = reward + gamma * np.max(Q[newState,:]) - Q[state, action]
#             Q[state, action] += alpha * err
#             errPerState[state].append(err)
                
#             state = newState
#             t += 1
            
#             if printSteps: 
#                 print("At time", t, ", we obtained reward", reward, ", and visited:", newState, "\n")
#                 # print("Next action:", actions[action])
            
#             if done:
#                 if printSteps: print("Episode finished" )
#                 break
        
#         env.close()
#     return Q, errPerState



# ### SARSA: ###
# #after Sutton & Barto, p130

# def SARSA(nrEpisodes, alpha, gamma, epsilon, printSteps=True):
#     errPerStateAction = {(i,a):list() for i in range(nrStates) for a in range(4)} # keeps track of error in each state.
#     for n in range(nrEpisodes):
#         env.reset() # Reset environment every episode?
#         t = 0
        
#         #state-action initialisation (action choice is epsilon-greedy):
#         state = 0
#         if np.random.rand() >= epsilon: 
#             action = np.argmax(Q[state,:]) 
#         else: 
#             action = env.action_space.sample()
        
#         # Run the game
#         while True:
#             if printSteps: env.render()
            
#             newState, reward, done, info = env.step(action)
            
#             #Choose action using eps-greedy policy from Q
#             if np.random.rand() >= epsilon: # with probability 1-epsilon, choose greedily
#                 newAction = np.argmax(Q[newState,:]) 
#             else: # with probability epsilon, do not choose greedy
#                 newAction = env.action_space.sample() #chooses random action
            
#             # Take chosen action, visit new state and obtain reward
            
#             # Update Q and save error to state:
#             err = reward + gamma * Q[newState,newAction] - Q[state, action]
#             Q[state, action] += alpha * err
#             errPerStateAction[state,action].append(err)
                
#             state = newState
#             action = newAction
#             t += 1
            
#             if printSteps: 
#                 print("At time", t, ", we obtained reward", reward, ", and visited:", newState, "\n")
#                 # print("Next action:", actions[action])
            
#             if done:
#                 if printSteps: print("Episode finished" )
#                 break
        
#         env.close()
#     return Q, errPerStateAction




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


# Summary
def printGameSummary(durations):
    print(evaluationMethod,":")
    print(f"Percentage of games won: {len(durations)/nrEpisodes*100}")
    print(f"Average duration winning game: {np.mean(durations)} steps")

### Execution ###
createEnv()

nrEpisodes = 5000
alpha = 0.1 # stepsize
gamma = 0.1 # discounting rate; may differ per state and iteration
eps = 0.1

def runSimulation(evaluationMethod):
    Q, errors, durations, gamesWon = policyEvaluation(nrEpisodes, alpha, gamma, evaluationMethod = "SARSA", epsilon = eps, printSteps = False)
    plotFromDict(errors, evaluationMethod)
    plotLearningCurve(gamesWon, evaluationMethod)
    printGameSummary(durations)


# TD: 
evaluationMethod = "TD"
runSimulation(evaluationMethod)

# Q-learning:
evaluationMethod = "Q"
runSimulation(evaluationMethod)

# SARSA:
evaluationMethod = "SARSA"
runSimulation(evaluationMethod)




### Currently unused: ###

#Could also opt for actual square representation?
def getCoordinate(index, gridsize):
    '''Takes location index and turns it into grid coordinate '''
    return (index // gridsize, index % gridsize) # // is floor division

# V = np.zeros((4,4)) #book says to initialise arbitrarily except for terminal states (for which V is 0)
# returns = np.zeros((4,4))
# d = np.zeros((4,4))