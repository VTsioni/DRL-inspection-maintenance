# Creating the basic class that will be used for the Benchmarks & PPO

# Importing basic libraries
import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
import seaborn as sns
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from tqdm import tqdm
import datetime
import time


class TrussBridgeEnv(gym.Env):
    def __init__(self, cost_dictionary, state_edges, transition_matrix, n_steps=70):

        # # Read costs from the costs dictionary
        self.cost_act = np.array([val['action_cost'] for val in cost_dictionary.values()])
        self.cost_mob = np.array([val['mobilisation_cost'] for val in cost_dictionary.values()])[0]

        # Number of components
        self.n_comps = len(list(cost_dictionary))

        # Number of states
        self.n_states = len(state_edges) - 1

        # Number of actions
        self.n_actions = 6

        # Define the action & observation space
        self.action_space = spaces.Discrete(self.n_actions * self.n_comps)
        self.observation_space = spaces.Discrete(self.n_states * self.n_comps)

        # Number of time steps
        self.n_steps = n_steps

        # Table that keeps the deterioration rates for all the components - 2D (components, t)
        self.time = np.tile(np.arange(self.n_steps), (self.n_comps, 1))

        # Transition probabilities - 3D matrix (t, s, s')
        self.trans_mat = transition_matrix.T

        # Read probabilities of failure for every state for every component from the dictionary
        self.Pf_state = np.load("C:\\Users\\vasts\\Documents\\AI project\\AInew\\pf_s.npy")

        # Initialising a (1, 70) table for the probability of failure of the system for each time step
        self.Pf_system_timeline = np.zeros(self.n_steps)

        # Initialising a (70, components) table for the probability of failure for every component at each time step
        self.Pf_comp_timeline = np.zeros((self.n_steps, self.n_comps))

        # Reset the environment
        self.reset()

    # Shifting the state of a particular component certain states back
    def _shift_states(self, comp_idx, shift=1):
        self.states[comp_idx, :] = np.roll(self.states[comp_idx, :], -shift)
        self.states[comp_idx, -1] = 0
        self.states[comp_idx, 0] = 1 - np.sum(self.states[comp_idx, 1:])
        self.states[comp_idx, :] = np.maximum(self.states[comp_idx, :], 0)
        self.states[comp_idx, :] /= self.states[comp_idx, :].sum()
        return

    # Changing the deterioration rate of a particular component according to a certain action
    def _change_deterioration_rate(self, comp_idx, action, steps=5):
        if action in [1, 4]:
            self.time[comp_idx, self.time_count:self.time_count + steps] = \
                np.minimum(self.time[comp_idx, self.time_count:self.time_count + steps], self.time_count)

        elif action in [2, 5]:
            self.time[comp_idx, self.time_count: self.time_count + steps] = 0
            self.time[comp_idx, self.time_count + steps:] = \
                np.arange(len(self.time[comp_idx, self.time_count + steps:])) + self.time[
                    comp_idx, self.time_count] + 1
        return

    # Because of certain actions we know exactly the state we are in
    def _determine_state(self, comp_idx, default_state=0):
        self.states[comp_idx, :] = 0
        self.states[comp_idx, default_state] = 1
        return

    # Perform the transition of states according to the transition probabilities
    def _transit_states(self, comp_idx):
        self.states[comp_idx] = np.matmul(self.states[comp_idx],
                                          self.trans_mat[self.time[comp_idx, self.time_count]])
        return

    # Creating the def step, to be called over one time step at a time
    # Inside the step function we go through for all the components and consider the relevant actions
    def step(self, action):

        done = False
        info = {}

        cost = 0

        # Creating a for loop for the number of components for one single time step
        for comp_idx in range(self.n_comps):
            current_action = action[comp_idx]

            if current_action in [0, 3]:  # no action
                # No change in deterioration rate
                pass

            elif current_action in [1, 4]:  # repair action
                # Each state shifts to the previous one
                self._shift_states(comp_idx=comp_idx, shift=1)
                # Deterioration rate stays the same for 5 years
                self._change_deterioration_rate(comp_idx=comp_idx, action=current_action, steps=5)

            elif current_action in [2, 5]:
                # Replacement means state goes back to 0
                self._determine_state(comp_idx=comp_idx, default_state=0)
                # Deterioration rate goes to 0 for 5 years and then starts again
                self._change_deterioration_rate(comp_idx=comp_idx, action=current_action, steps=5)

            elif current_action in [3, 4, 5]:
                # Inspection has taken place, so we sample an inspection outcome
                np.random.seed(1123895)
                observed_state = np.random.choice(range(self.states.shape[1]), 1,
                                                  p=self.states[comp_idx, :] / np.sum(self.states[comp_idx, :]))

                # self.states[comp_idx, :] = np.zeros_like(self.states[comp_idx, :])
                # self.states[comp_idx, :][obsState.astype(int)] = 1

                # Fix states distribution to the observed state
                self._determine_state(comp_idx, default_state=observed_state)

            # Update the states given the action that has been taken
            self._transit_states(comp_idx)

            # Compute the component cost and add it to the total step cost
            cost += self.cost_act[comp_idx, current_action]

        # Add a mobilisation cost once if an action has taken place for all the components
        mobil_cost = self.cost_mob[3] * np.any(current_action >= 3) + \
                     self.cost_mob[1] * np.any(np.c_[current_action == 1, current_action == 4]) + \
                     self.cost_mob[2] * np.any(np.c_[current_action == 2, current_action == 5])

        Pf_comp = np.matmul(self.states, self.Pf_state.T)
        Pf_system = 1 - np.prod(1 - Pf_comp)
        risk = Pf_system * 1e4

        cost += mobil_cost + risk

        # Check if this was the last iteration
        if self.time_count == self.time.shape[1]:
            done = True

        self.states_nn = np.concatenate([self.states.flatten(), self.time[:, self.time_count] / self.n_steps])

        # Update the time step
        self.time_count += 1

        return self.states_nn, cost, done, info

    def reset(self):
        # Reset the state and other parameters, if needed
        self.states = np.zeros((self.n_comps, self.n_states))
        self.states[:, 0] = 1

        self.time_count = 0
        self.time = np.tile(np.arange(self.n_steps), (self.n_comps, 1))
        self.states_nn = np.concatenate([self.states.flatten(), self.time[:, self.time_count] / self.n_steps])
        return

    def render(self):
        pass

    def close(self):
        pass
