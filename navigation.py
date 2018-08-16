from dqn import DQN
import Environment

env_name = 'visual_banana'

env = Environment.CollectBanana(env_name, 'Banana.x86_64')
dqn = DQN(env.name, env.state_size, env.action_size, env)
env.train_mode = True
dqn.train(target_score=13.0)
env.train_mode = False
dqn.play()


