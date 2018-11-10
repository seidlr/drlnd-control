# python test_agent.py --actor_model checkpoint_actor.pth --critic_model checkpoint_critic.pth
import argparse
import sys
import os

from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
import torch

from ddpg_agent import Agents

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the agent in the environment')
    parser.add_argument('--actor_model', required=True, help="path to the saved pytorch actor model")
    parser.add_argument('--critic_model', required=True, help="path to the saved pytorch critic model")

    result = parser.parse_args(sys.argv[1:])

    print (f"Selected actor model {result.actor_model}")
    
    if os.path.isfile(result.actor_model):
        print ("Actor model exists")
    else: 
        print ("Actor model not found")

    print (f"Selected critic model {result.critic_model}")
    if os.path.isfile(result.critic_model):
        print ("Critic model exists")
    else: 
        print ("Critic model not found")

    
    env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # number of actions
    action_size = brain.vector_action_space_size
    print('Size of each actions:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    # print('States look like:', state)
    state_size = states.shape[1]
    print('States have length:', state_size)

    agents = Agents(state_size=state_size, 
                  action_size=action_size, 
                  num_agents=num_agents, 
                  random_seed=0)

    agents.actor_local.load_state_dict(torch.load(result.actor_model))
    agents.critic_local.load_state_dict(torch.load(result.critic_model))

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations               # get the current state
    score = np.zeros(num_agents)                                          # initialize the score

    print ("Evaluating agents...")
    while True:
        action = agents.act(state) # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations      # get the next state
        rewards = env_info.rewards                     # get the reward
        dones = env_info.local_done                    # see if episode has finished
        agents.step(state, action, rewards, next_state, dones)
        score += rewards                               # update the score
        state = next_state                             # roll over the state to next time step
        if np.any(dones):                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(np.mean(score)))