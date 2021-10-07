# Calculating the deterioration of one element

# Importing basic libraries
import numpy as np
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
    LSF = yield_samples[:, None] - load_samples[:, None] * load_factor * (1/(Ao * (1-state)))
    return LSF

# Performing the Monte-Carlo analysis for a Gamma Distribution

if __name__ == '__main__':

    beta = 1.5

    tau_grid = np.arange(start=0, stop=70, step=1)
    det_rate = ExposureTime(tau=tau_grid, beta=beta)
    lam = CalculateLam(dm=0.4, std=0.075)
    shape, scale = GammaShapeScale(det_rate=det_rate, lam=lam)

    sample = np.random.gamma(shape=(det_rate[1:]-det_rate[:-1]), scale=scale, size=(1_000, len(tau_grid)-1))
    deterioration = np.cumsum(sample, axis=1)

    # Plotting the gamma process for the steel cross section loss due to corrosion

    fig = plt.figure()
    sns.set(style='white', font_scale=2)
    for realization in deterioration:
        plt.plot(tau_grid[1:], realization, c='k', alpha=0.3)

    plt.plot(tau_grid, det_rate/lam, c='r', label='Mean')
    plt.plot(tau_grid, det_rate/lam + np.sqrt(det_rate)/lam, c='r', linestyle='--', label='Mean+σ')
    plt.plot(tau_grid, det_rate/lam - np.sqrt(det_rate)/lam, c='r', linestyle='--', label='Mean-σ')

    plt.xlabel('Deterioration rate [years]')
    plt.ylabel('Section loss[-]')
    plt.legend()

    # Performing Monte-Carlo analysis to get the transition probabilities
    # for every deterioration rate for a single component

    # Defining state edges
    # Discretized space with a step of 2.5% section loss
    # if the section loss of a member exceeds 60% is considered failed

    state_edges = np.append(np.arange(start=0, stop=.61, step=.025), 1)

    n_states = len(state_edges) - 1

    # initialisation a 3D matrix

    trans_mat = np.zeros((n_states, n_states, len(tau_grid)-1))

    # conducting a for loop to get the transition probabilities for every time step

    # for time_index in range(sample.shape[1]-1):
    for time_index, _ in enumerate(deterioration[:, :-1].T):
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


    # Calculating the failure probability of every component of the structure


    P_states = np.zeros((trans_mat.shape[2], n_states))
    P_states[0, 0] = 1
    for time_index in range(trans_mat.shape[2]-1):
        P_states[time_index+1, :] = np.dot(P_states[time_index, :], trans_mat[:,:,time_index])

    # monte carlo for structural check
    load_samples= 100 + np.random.randn(1_000) * (0.1 * 100)
    yield_samples= (410 + np.random.randn(1_000) * 53.1)*10**3

    LSF_stress = calc_LSF(state=state_edges[:-1], load_samples=load_samples,
                          yield_samples=yield_samples, load_factor=3, Ao=900/1000**2)
    Pf_stress_cond_state = np.sum(LSF_stress<0, axis=0) / LSF_stress.shape[0]
    Pf_cond_state = np.where(state_edges[:-1]<=0.6, Pf_stress_cond_state, 1)

    Pf_time = np.dot(Pf_cond_state, P_states.T)

    plt.plot(tau_grid[1:], Pf_time)