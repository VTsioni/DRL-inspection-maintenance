# Creating the TCBM Benchmark for all the possible intervals & thresholds

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

# Input from probability_failure_elements, truss_environment
from probability_failure_elements import ExposureTime, \
    CalculateLam, GammaShapeScale, plotDetRate, CalculationTransitionMatrix
from steel_bridge_environment import TrussBridgeEnv

# Costs Dictionary
Investment = np.ones(11) * 100
Do_nothing = np.zeros(11)
Repair = np.ones(11) * 60
Replace = np.ones(11) * 130
Inspect = np.ones(11) * 50
Mobilisation = [0, 50, 50, 50, 100, 100]
Action = [100, 160, 230, 150, 210, 280]

cost_dictionary = {i: {'action_cost': Action, 'mobilisation_cost': Mobilisation} for i in range(len(Do_nothing))}


def timeBasedBenchmark(policy):
    # load the realisations from the MC so that the transition probabilities can be calculated
    deterioration = np.load("C:\\Users\\vasts\\Documents\\AI project\\AInew\\realisationsMC_1e3.npy")

    state_edges = np.append(np.arange(start=0, stop=.61, step=.025), 1)

    transition_matrix = CalculationTransitionMatrix(deterioration, state_edges)

    env = TrussBridgeEnv(cost_dictionary=cost_dictionary, state_edges=state_edges,
                         transition_matrix=transition_matrix, n_steps=70)

    state_space = np.arange(0, env.n_states)
    mean_state = np.dot(env.states, state_space.T)

    scenario_cost = 0

    action_log = np.ones((env.n_steps, env.n_comps))

    # Iterate over the lifecycle
    for i, timestep in enumerate(range(env.n_steps)):

        action = [0] * env.n_comps

        for comp in range(env.n_comps):

            # If nothing is selected, do nothing
            if mean_state[comp] >= repairThres:  # If its time to do a repair
                action[comp] = 1

            if mean_state[comp] >= replaceThres:  # If its time to do a replacement (replacement precedes repair)
                action[comp] = 2

            if (i + 1) % inspectInt == 0 and i != 0:  # # If its time to do an inspection
                action[comp] += 3

        action_log[i, :] = action

        new_states, cost, done, info = env.step(action)
        mean_state = np.dot(new_states, state_space.T)

        scenario_cost += cost

    return scenario_cost, action_log
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    intervalsInspect = np.arange(1, 71)
    thresholdsRepair = np.arange(1, 21)
    thresholdsReplace = np.arange(1, 21)
    cost_mat = np.ones((intervalsInspect.shape[0], thresholdsRepair.shape[0], thresholdsReplace.shape[0])) * np.nan
    actions_mat = np.ones((intervalsInspect.shape[0], thresholdsRepair.shape[0], thresholdsReplace.shape[0], 70, 11)) * np.nan

    for inspectIdx, inspectInt in tqdm(enumerate(intervalsInspect)):
        for replaceIdx, replaceThres in enumerate(thresholdsReplace):
            for repairIdx, repairThres in enumerate(thresholdsRepair[:replaceIdx]):
                time.sleep(0.01)
                policy = {'inspection_interval': inspectInt,
                          'repair_threshold': repairThres,
                          'replace_threshold': replaceThres}
                cost, actions_taken = timeBasedBenchmark(policy=policy)
                cost_mat[inspectIdx, repairIdx, replaceIdx] = cost
                actions_mat[inspectIdx, repairIdx, replaceIdx, :, :] = actions_taken

    indx_opt = np.unravel_index(np.nanargmin(cost_mat, axis=None), cost_mat.shape)
    opt_inspect = intervalsInspect[indx_opt[0]]
    opt_repair = thresholdsRepair[indx_opt[1]]
    opt_replace = thresholdsReplace[indx_opt[2]]
    opt_actions = actions_mat[indx_opt[0], indx_opt[1], indx_opt[2], :, :]
    opt_cost = cost_mat[indx_opt[0], indx_opt[1], indx_opt[2]]

    print("Optimal Cost: ", round(opt_cost, 2))
    print("Optimal Repair Threshold: ", opt_repair)
    print("Optimal Replace Threshold: ", opt_replace)
    print("Optimal Inspection Interval: ", opt_inspect)