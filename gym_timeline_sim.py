# Importing basic libraries

# imports
import gym
from gym import error, spaces, utils
import numpy as np


class SteelBridgeEnv(gym.Env):
    def __init__(self, tp, costs, numSteps=20):
        super(SteelBridgeEnv, self).__init__()
        # define the actions space
        # 3 actions * 2 inspection actions = 6 actions
        self.action_space = spaces.Discrete(6)

        # 4 different states:
        # 0: New, 1-19: Damages, 20: Failure
        self.observation_space = spaces.Discrete(21)

        self.n_states = self.observation_space.n

        # Number of steps
        self.numSteps = numSteps
        # Transition probabilities - 3D matrix (t, s, s')
        self.tp = tp

        self.time = np.arange(self.numSteps)
        self.time_count = 0

        self.costs = costs

        self.components = list(self.costs.keys())
        self.n_comp = len(self.components)

        self.Pf_state = np.array([val['Pf_state'] for val in self.costs.values()])

        self.Pf_system_timeline = np.zeros(self.numSteps)
        self.Pf_comp_timeline = np.zeros((self.numSteps,self.n_comp))

        # Reset the environment
        self.reset()

    def step(self, action):
        # import ipdb; ipdb.set_trace()
        # Auxiliary variables initialization
        done = False
        hasInsp = False
        info = {}

        Pf_comp = np.zeros(self.n_comp)
        cost = 0
        for comp_idx, comp in enumerate(self.components):
            # Calculate the new state distribution and the cost, based on the action given
            if action in [0, 3]:  # no action
                # Calculate the new state distribution with transition probabilities
                self.states[comp_idx, :] = np.dot(self.tp[self.time_count, :, :].T, self.states[comp_idx, :])
            Pf_comp[comp_idx] = np.dot(self.states[comp_idx, :], self.Pf_state[comp_idx,:])
            cost +=  self.costs[comp]['action'][0] + hasInsp * self.costs[comp]['inspect'][0]

        Pf_system = 1 - np.prod(1-Pf_comp)

        self.Pf_system_timeline[self.time_count] = Pf_system
        self.Pf_comp_timeline[self.time_count, :] = Pf_comp

        cost = cost + Pf_system * 1_000_000

        self.time_count += 1

        if self.time_count == len(self.time) - 1:
            done == True

        # Renormalize states probabilities!!!!
        self.states = self.states / np.sum(self.states)

        return self.states, cost, done, info

    def reset(self):
        # Reset the state and other parameters, if needed
        self.states = np.zeros((self.n_comp, self.observation_space.n))
        self.states[:, 0] = 1

    def render(self):
        pass

    def close(self):
        pass

#################################################################

pathTrans = "C:\\Users\\vasts\\PycharmProjects\\montecarlodeteriorationrate\\"# path for transition probabilities
patternTrans = "trans_mat_all2"# filename pattern for transition probabilities matrix

pathFail = "C:\\Users\\vasts\\PycharmProjects\\montecarlodeteriorationrate\\Risks\\" # path for failure probabilities
patternFail = "risk_failure_comp" # filename pattern for failure probabilities

pathCosts = "C:\\Users\\vasts\\PycharmProjects\\montecarlodeteriorationrate\\Costs\\" # path for costs
patternInspCost = "cost_insp_comp"
patternStateCost = "cost_state_comp" ## NOT IMPLEMENTED YET
patternActCost = "cost_act_comp"

#Number of steps
numSteps = 70
states = 21
actions = 6

### # Simple policy selector
#simplePol = 1

for simplePol in [1]:

    # Components
    comps = np.arange(1, 12) # Component ids from 01 to 11
    costs = {}
    tp = np.load(pathTrans + patternTrans + ".npy")

    for comp in comps:

        # Risk of failure probabilities for component comp
        riskFilename = pathFail + patternFail + str(comp).zfill(2) + ".npy"
        riskFail = np.load(riskFilename)

        # Costs
        c_act = np.tile(np.load(pathCosts + patternActCost + str(comp).zfill(2) + ".npy"), 2)
        c_ins = np.array([np.load(pathCosts + patternInspCost + str(comp).zfill(2) + ".npy")])
        # c_risk = riskFail * np.max(c_act)
        c_state = np.concatenate(([0], np.logspace(0, states, num=states, endpoint=True, base=1.2)[1:]))
        costs[comp]={'action': c_act, 'inspect': c_ins, 'state': c_state, 'Pf_state': riskFail}

    # Initiate environment object
    env = SteelBridgeEnv(tp, costs, numSteps=numSteps)
    cost_list = []
    new_states = env.states
    for i in range(env.numSteps):

        # Initializations
        if simplePol == 1:
            # Policy 1: Do nothing
            action = 0
        elif simplePol == 2:
            # Policy 2: Replace when in last state
            if new_states[-1] >= 0.5:
                action = 2
            else:
                action = 0
        elif simplePol == 3:
            # Policy 3: Perform an inspection when there is no dominant probability state
            domState = np.argmax(new_states)
            if np.max(new_states) <= 0.5:
                if domState >= 0 and domState <= 10:
                    action = 3
                elif domState > 10 and domState <= 19:
                    action = 4
                else:
                    action = 5
            else:
                if domState >= 0 and domState <= 10:
                    action = 0
                elif domState > 10 and domState <= 19:
                    action = 1
                else:
                    action = 2
        new_states, cost, done, info = env.step(action)

        cost_list.append(cost)
        # compCosts.append(totCost)
        print("Iteration {}: Action {}, States {}, Cost {}".format(str(i), str(action), ','.join(str(state) for state in new_states), str(cost)))


cost_list = np.array([item.squeeze() for item in cost_list])

import matplotlib.pyplot as plt
plt.plot(env.Pf_system_timeline)
plt.plot([item.squeeze() for item in cost_list])