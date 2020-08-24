import gym
import numpy as np
import time
import matplotlib

env = gym.make("MountainCar-v0")

# print(env.reset()) # tuple(position, velocity)
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)
# print(env._max_episode_steps)

bin_size = (20,20)
learning_rate = 0.1
discount_rate = 0.95
episodes = 25000
epsilon = 0.5
epsilon_start_decay = 1
epsilon_end_decay = episodes // 2
epsilone_decay_value = epsilon / (epsilon_end_decay - epsilon_start_decay)
print("Shape of q table", qshape)

def discrete(state):
    binidxs = (state - env.observation_space.low) / ((env.observation_space.high - env.observation_space.low)/bin_size)
    return tuple(binidxs.astype(np.int))
qshape = bin_size + (env.action_space.n,)
qtable = np.random.uniform(low=-1,high=0, size=qshape)
# qtable.tofile(f"qtable.0.txt")
qtable = np.fromfile("qtable.final.txt", dtype=qtable.dtype).reshape(qshape)
for ep in range(episodes):
    if ep % 5000 == 0:
        render = True
    else:
        render = False
    steps = 0
    discrete_state = discrete(env.reset())
    while True:
        if np.random.random() > epsilon:
            action = np.argmax(qtable[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        if epsilon_end_decay >= ep >=  epsilon_start_decay:
            epsilon -= epsilone_decay_value
        new_state, reward, done, info = env.step(action)
        new_discrete_state = discrete(new_state)
        steps += 1
        if render:
            env.render()
        if not done: # do q-learning
            max_future_q = np.max(qtable[new_discrete_state])
            current_q = qtable[discrete_state + (action,)]
            new_q = (1-learning_rate)*current_q + learning_rate*(reward + discount_rate*max_future_q)
            qtable[discrete_state + (action,)] = new_q
            discrete_state = new_discrete_state
        else:
            if new_state[0] >= env.goal_position:
                # reach the goal within 200 steps, reward
                qtable[discrete_state + (action,)] = 0
                if ep % 1000 == 0:
                    print(f"reach goal at {ep} with total {steps} steps")
                if steps <= 89:
                    print(f"reach goal at {ep} with total {steps} steps")
            # else:
            #     print("Failed to finish the game")
            break
qtable.tofile(f"qtable.final.txt")
env.close()



