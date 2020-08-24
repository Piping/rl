import gym
import numpy as np
import time
import readchar

env = gym.make("MountainCar-v0")
state = env.reset()

print(state) # tuple(position, velocity)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

env.render()

def input_to_action():
    char = readchar.readchar()
    if char == 'a':
        return 0
    elif char == 's':
        return 1
    elif char == 'd':
        return 2
    else:
        return 1

num_steps = 0

while True:
    action = input_to_action()
    new_state, reward, done, debug_info = env.step(action)
    env.render()
    num_steps += 1
    if done:
        break
env.close()

print(f'You used {num_steps} steps to finish the game')



