# Super Mario Bros: Reinforcement Learning (SMRL)

Welcome to my COMP 579 final project on reinforcement learning for Super Mario Bros.

To run agents, execute smrl_env.py. In that file, a deep RL agent can be selected by changing the 'agent' object on line 86 (there are 4 options: DqnAgent, DdqnAgent, DuelingDqnAgent, PpoAgent), the 'agent type' on line 88, and whether an existing model is loaded and which one. 

Note that only the PPO and Dueling DQN have existing trained models. These range from being trained 1000 to 60000 episodes with a step of 1000 episodes. Be sure to select an appropriate number of episodes on line 97
