from unityagents import UnityEnvironment
import numpy as np

class CollectBanana():
    def __init__(self, name, file_name):
        self.name = name
        self.base = UnityEnvironment(name+'/'+file_name)
        # get the default brain
        self.brain_name = self.base.brain_names[0]
        self.brain = self.base.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.train_mode = True
        self.reset()
        if name == 'visual_banana':
            self.state_size = self.state.shape
        else:
            self.state_size = len(self.state)

    def get_state(self):
        if self.name == 'visual_banana':
            # state size is 1,84,84,3   0123-0312
            # Rearrange from NHWC to NCHW
            self.state = np.transpose(self.env_info.visual_observations[0], (0,3,1,2))
        else:
            self.state = self.env_info.vector_observations[0]

    def reset(self):
        self.env_info = self.base.reset(train_mode=self.train_mode)[self.brain_name]
        self.get_state()
        return self.state

    def render(self):
        pass

    def step(self, action):
        self.env_info = self.base.step(action)[self.brain_name]  # send the action to the environment
        self.get_state()
        reward = self.env_info.rewards[0]  # get the reward
        done = self.env_info.local_done[0]  # see if episode has finished
        return self.state, reward, done, None #info is none

    def close(self):
        self.base.close()
