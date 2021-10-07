# Calculating the tree of failure for the system of 11 components

# Importing basic libraries
import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
import seaborn as sns
import matplotlib.pyplot as plt

# Defining the primary functions
# we consider failure of the component at 40% cross-section loss,
# std=7.5%, for 70 years life-cycle
# f(τ)/λ = 0.40 & f(τ) = α * τ^β

def ExposureTime(tau, alpha=28.44/70**1.5, beta=1):
    return alpha * tau**beta

def GammaShapeScale(det_rate, lam):
    shape = det_rate
    scale = 1 / lam
    return shape, scale

def CalculateLam(dm, std):
    lam = dm / std**2
    return lam

def calc_LSF(state, load_samples, yield_samples, Ao, load_factor):
    LSF = yield_samples[:, None] * Ao * (1-state) / (load_samples[:, None] * load_factor) - 1
    return LSF

def CalculationTransitionMatrix(deterioration, state_edges):

    n_states = len(state_edges) - 1

    # initialisation a 3D matrix

    trans_mat = np.zeros((n_states, n_states, deterioration.shape[1] - 1))

    # conducting a for loop to get the transition probabilities for every time step

    for time_index in range(deterioration.shape[1] - 1):
        time_sample_prev = deterioration[:, time_index]
        time_sample_next = deterioration[:, time_index + 1]

        index_prev = np.digitize(x=time_sample_prev, bins=state_edges, right=False)
        index_next = np.digitize(x=time_sample_next, bins=state_edges, right=False)

        # trans_mat[:, :, time_index] = np.eye(n_states)
        for i, state_prev in enumerate(state_edges[1:]):
            Ni = np.sum(np.where(index_prev == i + 1, 1, 0))
            if Ni != 0:
                for j, state_next in enumerate(state_edges[1:]):
                    Nij = np.sum(np.where(np.all(np.c_[index_prev == i + 1, index_next == j + 1], axis=1), 1, 0))
                    trans_mat[i, j, time_index] = Nij / Ni
    return trans_mat

    # Calculating the probability of being in each state for all time steps

def CalcProbabilityState(trans_mat, state_edges):

    n_states = len(state_edges) - 1
    P_states = np.zeros((trans_mat.shape[2], n_states))
    P_states[0, 0] = 1
    for time_index in range(trans_mat.shape[2] - 1):
        P_states[time_index + 1, :] = np.dot(P_states[time_index, :], trans_mat[:, :, time_index])

    return P_states

# Performing Monte Carlo for the structural check of a component

def StructuralCheckMC(distribution_dict, state, load_factor, Ao, n_mc=1_000):
    load_samples = distribution_dict['load']['mean'] + \
                   np.random.randn(n_mc) * distribution_dict['load']['std']

    yield_samples = distribution_dict['yield']['mean'] + \
                    np.random.randn(n_mc) * distribution_dict['yield']['std']

    LSF_stress = calc_LSF(state=state, load_samples=load_samples,
                          yield_samples=yield_samples, load_factor=load_factor, Ao=Ao)
    return LSF_stress


def plotDetRate(deterioration, tau_grid, det_rate, lam):
    fig = plt.figure()
    sns.set(style='white', font_scale=2)
    for realization in deterioration:
        plt.plot(tau_grid, realization, c='k', alpha=0.3)

    plt.plot(tau_grid, det_rate/lam, c='r', label='Mean')
    plt.plot(tau_grid, det_rate/lam + np.sqrt(det_rate)/lam, c='r', linestyle='--', label='Mean+σ')
    plt.plot(tau_grid, det_rate/lam - np.sqrt(det_rate)/lam, c='r', linestyle='--', label='Mean-σ')

    plt.xlabel('Deterioration rate [years]')
    plt.ylabel('Section loss[-]')
    plt.legend()

def plotPftime(Pf_comp, Pf_system):
    fig = plt.figure()
    for comp in Pf_comp:
        plt.plot(comp, c='k', alpha=0.5)
    plt.plot(Pf_system, c='r', zorder=11111)

    plt.xlabel('Deterioration rate [years]')
    plt.ylabel('Probability of Failure [annual]')
    plt.legend()


if __name__ == '__main__':

    # Performing the Monte-Carlo analysis for a Gamma Distribution

    beta = 1.5

    tau_grid = np.arange(start=0, stop=71, step=1)
    det_rate = ExposureTime(tau=tau_grid, beta=beta)
    lam = CalculateLam(dm=0.4, std=0.075)
    shape, scale = GammaShapeScale(det_rate=det_rate, lam=lam)

    sample = np.random.gamma(shape=(det_rate[1:] - det_rate[:-1]), scale=scale, size=(1_000, len(tau_grid) - 1))
    deterioration = np.c_[np.zeros((sample.shape[0], 1)), np.cumsum(sample, axis=1)]

    plotDetRate(deterioration=deterioration, tau_grid=tau_grid, det_rate=det_rate, lam=lam)


    # Defining state edges
    # Discretized space with a step of 2.5% section loss
    # if the section loss of a member exceeds 60% is considered failed

    state_edges = np.append(np.arange(start=0, stop=.61, step=.025), 1)

    # Calling the def for the calculation of the transition matrix
    transition_matrix = CalculationTransitionMatrix(deterioration, state_edges)

    # Calling the def for the calculation of the probability of being in each state for all time steps
    p_state = CalcProbabilityState(transition_matrix, state_edges)


    # Creating a dictionary for the values of the load and yield of the elements
    distribution_dict = {'load':{'mean':100, 'std':0.1*100}, 'yield':{'mean':410*10**3, 'std':53.1*10**3}}

    # Creating a for loop for all the elements of the system

    load_factor = [3, 3, 3, 3, 5, 2, 5/3, 5/3, 2, 5, 5+ 1/3]
    Ao = [1_300, 1_300, 1_300, 1_300, 2_200, 900, 900, 900, 900, 2_200, 2_200]

    element_dict = {i: {'load_factor': lf, 'Ao':ao*(10**(-6))} for i, (lf, ao) in enumerate(zip(load_factor, Ao))}

    states = state_edges[:-1]

    # Probability of failure in every state
    # Intialization with an empty list
    Pf_s = []

    for key in element_dict.keys():
        value = element_dict[key]

        LSF_stress = StructuralCheckMC(distribution_dict=distribution_dict, state=states,
                                       load_factor=value['load_factor'], Ao=value['Ao'], n_mc=1_000)

        Pf_stress_cond_state = np.sum(LSF_stress < 0, axis=0) / LSF_stress.shape[0]
        Pf_cond_state = np.where(states <= 0.6, Pf_stress_cond_state, 1)
        Pf_s.append(Pf_cond_state)

    Pf_s = np.array(Pf_s)

    # Probability of failure per component at each time step =
    # probability of failure of every component in every state
    # given the probability of being in that state in every times step

    Pf_comp_time = np.dot(Pf_s, p_state.T)
    Pf_system_time = 1 - np.prod(1-Pf_comp_time, axis=0)

    plotPftime(Pf_comp=Pf_comp_time , Pf_system=Pf_system_time)

    df = pd.DataFrame(data=np.c_[tau_grid[1:], det_rate[1:], Pf_comp_time.T, Pf_system_time],
                      columns=['time', 'deterioration rate'] + ['Pf_comp_'+str(key) for key in element_dict.keys()] + ['Pf_system'])

    df.to_csv('Probability_Failure_System.csv', index=False)
    




