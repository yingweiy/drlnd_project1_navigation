import torch
import numpy as np
from collections import deque
from dqn_agent import Agent


class DQN():

    # env assumption: env.reset(), env.render(), env.step(), env.close()

    def __init__(self, name, state_size, action_size, env):
        self.agent = Agent(state_size=state_size, action_size=action_size, seed=0)
        self.env = env
        self.saved_network = name+'_dqn_checkpoint.pth'

    def train(self, n_episodes=2000, max_t=1000, eps_start=1.0,
              eps_end=0.01, eps_decay=0.995,
              score_window_size=100, target_score=13.0,
              save=True,
              verbose=True):
        """Deep Q-Learning.

            Params
            ======
                n_episodes (int): maximum number of training episodes
                max_t (int): maximum number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=score_window_size)  # last score_window_size scores
        eps = eps_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            if np.mean(scores_window) >= target_score:
                if verbose:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                self.solved = True
                if save:
                    torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)
                break

            if verbose:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if save:
            torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)

        return scores

    def play(self, trials=3, steps=200, load=False):
        if load:
            self.agent.qnetwork_local.load_state_dict(torch.load(self.saved_network))

        for i in range(trials):
            state = self.env.reset()
            for j in range(steps):
                action = self.agent.act(state)
                self.env.render()
                state, reward, done, _ = self.env.step(action)
                if done:
                    break
        self.env.close()
