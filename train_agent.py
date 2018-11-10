# python train_agent.py --episodes 1000 --model checkpoint --plot Score.png
import argparse
from collections import deque
import datetime
import sys
import time
import os

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ddpg_agent import Agents

def ddpg(n_episodes=2000, max_t=1000, store_model='checkpoint'):
    """DDPG-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        save_model (str): path for storing pytoch model
    """
    start = time.time()

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations  
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agents.act(state)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agents.step(state, action, rewards, next_state, dones)
            state = next_state
            score += rewards
            if np.any(dones):
                print('\tSteps: ', t)
                break 
        scores_window.append(np.mean(score))       # save most recent score
        scores.append(np.mean(score))              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.3f}\t{}'.format(i_episode, 
                                                           np.mean(scores_window),
                                                           np.mean(score),
                                                           datetime.datetime.now()), end='')
        average_score = np.mean(scores_window)
        if i_episode % 25 == 0 or average_score > 30:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
            torch.save(agents.actor_local.state_dict(), f'{store_model}_actor.pth')
            torch.save(agents.critic_local.state_dict(), f'{store_model}_critic.pth')
            
        if average_score>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, average_score))
            end_ = time.time()
            print("\nTime needed for solution: {} s".format(end_ - start))
            break
    return scores

def plot_scores(scores, rolling_window=10, save_plot='Score.png'):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean, linewidth=4);
    plt.savefig('Score.png')
    return rolling_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the agent in the environment')
    parser.add_argument('--episodes', help="How many episodes to train the agent")
    parser.add_argument('--model', default='checkpoint.pth', help="path where the pytorch model should be stored")
    parser.add_argument('--plot', help="path to save the achieved training score of the agent")

    options = parser.parse_args(sys.argv[1:])
    
    env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64")
    #env = UnityEnvironment(file_name="Reacher_Linux_Single/Reacher.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print("Using {}".format(brain_name))

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

    scores = ddpg(n_episodes=int(options.episodes), store_model=options.model)

    plot_scores(scores, rolling_window=10, save_plot=options.plot)
    