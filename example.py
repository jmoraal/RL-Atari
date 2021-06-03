import gym

env = gym.make ("Breakout-v0")

print(env.action_space)
print(env.observation_space)

print("Press Enter to cont inue . . . " )
input()

env.reset()

# Run the game
for t in range (1000):
    env.render()
    
    # Choose a random action
    action = env.action_space.sample()
    
    # Take the action , make an observation from environment and obtain reward
    observation, reward, done, info = env.step(action)
    print ("At time " , t , " , we obtained reward " , reward , " , and observed : " )
    print (observation)
    
    if done:
            print ("Episode finished " )
            break

env.close()