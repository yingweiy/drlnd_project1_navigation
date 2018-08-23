from dqn import DQN
import Environment
import numpy as np

env_name = 'visual_banana'
#env_name = 'basic_banana'

env = Environment.CollectBanana(env_name, 'Banana.x86_64')
dqn = DQN(env.name, env.state_size, env.action_size, env)
env.train_mode = True
scores = dqn.train(n_episodes=2000, target_score=13.0)
np.save('scores.npy', np.array(scores))
env.train_mode = False
dqn.play(load=True)


