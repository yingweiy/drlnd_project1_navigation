from dqn import DQN
from unityagents import UnityEnvironment

class Environment():
    def __init__(self, name):
        self.name = name
        self.base = UnityEnvironment(file_name="Banana.x86_64")
        # get the default brain
        self.brain_name = self.base.brain_names[0]
        self.brain = self.base.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.train_mode = True
        self.reset()
        self.state_size = len(self.state)


    def reset(self):
        self.env_info = self.base.reset(train_mode=self.train_mode)[self.brain_name]
        self.state = self.env_info.vector_observations[0]
        return self.state

    def render(self):
        pass

    def step(self, action):
        self.env_info = self.base.step(action)[self.brain_name]  # send the action to the environment
        next_state = self.env_info.vector_observations[0]  # get the next state
        reward = self.env_info.rewards[0]  # get the reward
        done = self.env_info.local_done[0]  # see if episode has finished
        self.state = next_state
        return next_state, reward, done, None #info is none

    def close(self):
        self.base.close()


env = Environment('BananaNavigation')
dqn = DQN(env.name, env.state_size, env.action_size, env)
env.train_mode = True
dqn.train()
env.train_mode = False
dqn.play()


