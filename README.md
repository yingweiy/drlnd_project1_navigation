[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

### Udacity Deep Reinforcment Learning Nanodegree 
# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Files and Folders
* navigation.py: the main routine for the basic-banana project. It is the high-level calls to construct the environment, agent, train and play. 
* visual_navigation.py: the main routine for the challenge project - visual banana.
* dqn.py: the deep reinforcement learning algorithm routine.
* dqn_agent.py: the class definitions of the agent and replay buffer.
* model.py: the deep neural network models are defined in this file.
* Environment.py: a wrapper for the UnityEnvironment. The wrapper makes the 
environment interface similar to the OpenAI gym environment, so that 
the DQN routines are more general.
* dqn_test.py: a dqn test routine using the OpenAI gym environment (can be ignored).
* folder ``basic_banana``: the directory to save the unity environment run time of the basic agent in
 sensor mode.
* folder ``visual_banana``: the directory to save the unity environment run time of the agent in
 pixel/visual mode.
* PlotScores.ipynb: the routine to plot the scores.
 
#### Environment Wrapper Class
The ``CollectBanana`` Class is a wrapper for the ``UnityEnvironment`` class, which 
includes the following main methods:
* step
* reset
* get_state
* close

The ``name`` parameter in the 
constructor allows the selection of ``state`` format returned:
* For basic banana, the state is a 37-dimensional vector.
```self.state = self.env_info.vector_observations[0]```

* For visual banana, the state contains four frames by calling ``get_state()``. 
To fit the PyTorch format, the original frame format is transposed from NHWC (Batch, Height, Width, Channels) to NCHW 
 by numpy transpose function as follows:
``frame = np.transpose(self.env_info.visual_observations[0], (0,3,1,2))`` 
The current frame, together with previous frame ``last_frame`` and the second previous frame, ``last2_frame``
are then assemblied into variable ``CollectBanana.state`` variable.

#### DQN and Agent

The DQN used in this implementation is the simple DQN with two networks: one is local, and one is target.

In dqn_agent.py, the network is created with these lines:
```python
if qnetwork_type=='visual_banana':
    self.qnetwork_local = VisualQNetwork(state_size, action_size, seed).to(device)
    self.qnetwork_target = VisualQNetwork(state_size, action_size, seed).to(device)
else:
    self.qnetwork_local = BasicQNetwork(state_size, action_size, seed).to(device)
    self.qnetwork_target = BasicQNetwork(state_size, action_size, seed).to(device)
```

For both basic- and visual-banana project, the training parameters are as follows:

In the dqn.py: Class DQN train() method:
```python
n_episodes=2000, max_t=1000, eps_start=1.0,
eps_end=0.01, eps_decay=0.995,
score_window_size=100, target_score=13.0
```

In the dqn_agent.py file: the Agent class has parameters to learn():
```python
BUFFER_SIZE = int(10000)  # replay buffer size
BATCH_SIZE = 64         # minibatch size

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
```

#### Neural Network Models

The neural networks are defined in the model.py file. It contains two classes: 
* BasicQNetwork: it contains two fully connected layers. Each layer contains 128 neurons. 
```python
x = F.relu(self.fc1(state))
x = F.relu(self.fc2(x))
return self.fc3(x)
```
* VisualQNetwork
The visualQNetwork employs three 3D convolutional layers and 2 fully connected layers. It is defined as 
follows:
```python
    nfilters = [128, 128*2, 128*2]
    self.seed = torch.manual_seed(seed)
    self.conv1 = nn.Conv3d(3, nfilters[0], kernel_size=(1, 3, 3), stride=(1,3,3))
    self.bn1 = nn.BatchNorm3d(nfilters[0])
    self.conv2 = nn.Conv3d(nfilters[0], nfilters[1], kernel_size=(1, 3, 3), stride=(1,3,3))
    self.bn2 = nn.BatchNorm3d(nfilters[1])
    self.conv3 = nn.Conv3d(nfilters[1], nfilters[2], kernel_size=(4, 3, 3), stride=(1,3,3))
    self.bn3 = nn.BatchNorm3d(nfilters[2])
    conv_out_size = self._get_conv_out_size(state_size)
    fc = [conv_out_size, 1024]
    self.fc1 = nn.Linear(fc[0], fc[1])
    self.fc2 = nn.Linear(fc[1], action_size)
```


#### Results

##### Basic Banana
```angular2html
Episode 100	Average Score: 0.555
Episode 200	Average Score: 3.00
Episode 300	Average Score: 6.85
Episode 400	Average Score: 9.35
Episode 500	Average Score: 11.72
Episode 600	Average Score: 12.46
Episode 700	Average Score: 12.69
Episode 709	Average Score: 12.91
Environment solved in 710 episodes!	Average Score: 13.01
```
![alt text](./basic_banana_scores.png)

##### Visual Banana
```angular2html
Episode 100	Average Score: 0.133
Episode 200	Average Score: 0.89
Episode 300	Average Score: 2.66
Episode 400	Average Score: 5.50
Episode 500	Average Score: 8.63
Episode 600	Average Score: 9.60
Episode 700	Average Score: 11.16
Episode 800	Average Score: 11.36
Episode 900	Average Score: 12.13
Episode 984	Average Score: 12.94
Environment solved in 985 episodes!	Average Score: 13.03
```
![alt text](./visual_banana_scores.png)


#### Thoughts on the Challenge Project
From the challenge project, I observed the following:
* Color images are better than than the gray scale images, because the bananas are in yellow and blue, also 
the background wall, floor, and sky. It is hard to distinquish them with only gray pictures.
* Using multiple frames instead of single one. I used four frames and it improves the results a lot.
* Using 3D convolutional layers yields bettter results than 2D in my test. That perhaps because the 
input state are multiple images and temporal relationships. 3D filters are keen to detect the local
motion patterns.
* The number of filters cannot be too small. I used 128, 256, 256 in these layers, which are much 
better than small numbers that I tried earlier, such as 32, 64, 64. 

## Project Instruction from Udacity
### Getting Started

1. Install UNITY and ML-Agent following this instruction: 
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

To install Unity on Ubuntu, see this post:
https://forum.unity.com/threads/unity-on-linux-release-notes-and-known-issues.350256/page-2

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
