# Project 1: Navigation 

**Note: This is for project submission for the Udacity Deep Reinforcement Learning Nanodegree program.**


## Project Details

For this project, the network has been trained to navigate in a large, square world and collect bananas. The problem is set up so that a reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal is to maximize the reward.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.


## Getting Started

The windows inverontent is provided with the given repository. But you can dowload the coresponding environement matching your operating system.
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

**Note:** I have trained the network using CPU which required for me to explicitly indicate the CPU instead of the GPU in **dqn-agent.py**. If you want to use GPU uncomment the GPU availability line and comment the CPU assignment line. 
'''Python
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
'''


## Instructions

Project consists of 
 - **Report.ipynb** Where one will open the environment, train the agent and can see the trained agent play the game.
 - **dqn_agent.py** Which contains all the description of the agent.
 - **model.py** Containing network model. 

1. Start the environment
    Fist start the environment and make sure that it launches.
2. Examine the state and action spaces
    Print the available action space length and state vector associated to the given image. 
3. Take random actions in the environment
    You can interact with the environment by taking the random action out of available action space. 
4. Train the model
    The notebook contains the function which trains the agent for 2000 epochs, 1000 maximum number of timesteps per episode, 1.0 starting value of epsilon, 0.01 minimum value of epsilon and 0.995 multiplicative factor (per episode) for decreasing epsilon.
    Scores for each episode is plotted after the training. 
    The network was trained for 2000 episodes, but we can see that the network was able to resolve the environment within 1000 epochs.
    A plot of rewards per episode is included inside the notebook.
5. Take trained actions in the environment
    The trained model is provided. Thus you can skip the training phase and see how the trained agent performs the task. 
6. close the environment 
    Ones you are done with the environment you can close it from the notebook. 


## Ideas for Future Work

The further improvements can be achieved by manipulating the hyperparameters. For example, one can increase the number of time steps and change the epsilon multiplicative factor to 0.999. This will force the agent to value the future reward more and also to live longer. Thus increasing the overall score per episode. Also, the model can be improved by manipulating the network architecture and adding the dropout layers which will help to generalize the network better. 