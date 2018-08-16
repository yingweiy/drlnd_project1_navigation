from dqn import DQN
import gym

env = gym.make('LunarLander-v2')
env.seed(0)
dqn = DQN('lunar_landing', 8, 4, env)
dqn.train(target_score=200)
dqn.play(load=True)



