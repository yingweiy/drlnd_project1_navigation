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
* navigation.py: the high-level calls to construct the environment, agent, train and play. This is the main entrance.
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

* For visual banana, the state contains three frames by calling ``get_state()``. 
To fit the PyTorch format, the original frame format is transposed from NHWC (Batch, Height, Width, Channels) to NCHW 
 by numpy transpose function as follows:
``frame = np.transpose(self.env_info.visual_observations[0], (0,3,1,2))`` 
The current frame, together with previous frame ``last_frame`` and the second previous frame, ``last2_frame``
are then assemblied into variable ``CollectBanana.state`` variable.

#### DQN and Agent


#### Neural Network Models


#### Results

##### Basic Banana


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
