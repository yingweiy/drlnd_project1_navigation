from dqn import DQN
import matplotlib.pyplot as plt
import gym

env = gym.make('LunarLander-v2')
env.seed(0)
dqn = DQN('lunar_landing', 8, 4, env)
dqn.train()
dqn.play()



