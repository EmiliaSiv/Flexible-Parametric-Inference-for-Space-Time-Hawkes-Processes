# %% import stuff
# import libraries

import itertools
import pandas as pd
import time
import numpy as np
import torch
from joblib import Memory, Parallel, delayed

from spatio_temporal_fadin.utils.simulation import *
from spatio_temporal_fadin.utils.discretisation import smooth_projection_spatio_temp
from spatio_temporal_fadin.utils.compute_constants import precomp_phi_events_prepsi
from spatio_temporal_fadin.solver import SpatialFaDIn


################################
# Meta parameters
################################

spatio_bound_x = 10
spatio_bound_y = 10
K = [spatio_bound_x, spatio_bound_y]
spatio_bound = [-spatio_bound_x, -spatio_bound_y]
delta_spatio_x = 0.1
delta_spatio_y = 0.1
delta_spatio = [delta_spatio_x, delta_spatio_y]
Ws_x = 1
Ws_y = 1
Ws = [Ws_x, Ws_y]

delta_time = 0.1
T = 10
Wt = 1

delta = [delta_spatio, delta_time]
kernel = ["truncated_gaussian", "truncated_gaussian"]
kernel_length = [Ws, Wt]
grad_kernel = ["truncated_gaussian", "truncated_gaussian"]

mem = Memory(location=".", verbose=0)

# %% simulate data
# Simulated data
################################

mu = 0.5
alpha = 0.6
mean = 0.0
sigma = 0.1
mean_time = 0.5
sigma_time = 0.1

baseline_true_fixed = np.array([mu])
alpha_true_fixed = np.array([[alpha]])
kernel_params_space_x = [torch.tensor(([[mean]])), torch.tensor(([[sigma]]))]
kernel_params_space_y = [torch.tensor(([[mean]])), torch.tensor(([[sigma]]))]
kernel_params_time = [torch.tensor(([[mean_time]])), torch.tensor(([[sigma_time]]))]

def precomp_psi_loops(events_grid, L_S, L_T):
    n_dim, G_S_x, G_S_y, G_T = events_grid.shape
    res = torch.zeros(L_S[0], L_S[1], L_T, L_S[0], L_S[1], L_T)

    L_x = L_S[0]//2
    L_y = L_S[1]//2

    indx = 0
    for taux in range(-L_x, L_x+1):
        indy = 0
        for tauy in range(-L_y, L_y+1):
            for taut in range(L_T):
                indx2 = 0
                for taux2 in range(-L_x, L_x+1):
                    indy2 = 0
                    for tauy2 in range(-L_y, L_y+1):
                        for taut2 in range(L_T):
                            for vx in range(G_S_x):
                                for vy in range(G_S_y):
                                    for vt in range(G_T):
                                        if vx - taux >= 0 and vx - taux < G_S_x and vy - tauy >= 0 and vy - tauy < G_S_y:
                                            if vx - taux2 >= 0 and vx - taux2 < G_S_x and vy - tauy2 >= 0 and vy - tauy2 < G_S_y:
                                                if vt - taut >= 0 and vt - taut < G_T and vt - taut2 >= 0 and vt - taut2 < G_T:
                                                    res[indx, indy, taut, indx2, indy2, taut2] +=\
                                                        (events_grid[0, vx - taux, vy - tauy, vt - taut]\
                                                        * events_grid[0, vx - taux2, vy - tauy2, vt - taut2])
                        indy2 += 1
                    indx2 += 1
            indy += 1
        indx += 1
    return res

@mem.cache
def simulate_data(T, K, Wt, Ws, baseline, alpha, kernel_params_time,
                         kernel_params_space_x, kernel_params_space_y, seed=0):

    Kx = K[0]
    Ky = K[1]

    support = ((-Kx, Kx), (-Ky, Ky))
    end_time = T

    time_kernel = 'norm'
    space_x_kernel = 'norm'
    space_y_kernel = 'norm'
    
    events, immigrants = simu_spatial_hawkes_cluster_custom(end_time, support,
                            baseline, alpha, time_kernel, space_x_kernel,
                            space_y_kernel, kernel_params_time,
                            kernel_params_space_x, kernel_params_space_y, Wt, Ws, seed)

    return events

@mem.cache
def run_matrices(events, T, K, spatio_bound, kernel_length, delta, seed=0):

    start = time.time()

    delta_s = delta[0]
    delta_t = delta[1]
    W_s = kernel_length[0]
    W_t = kernel_length[1]

    n_grid_spatio_x = int(2 * K[0] / delta_s[0]) + 1
    n_grid_spatio_y = int(2 * K[1] / delta_s[1]) + 1
    n_grid_spatio = [n_grid_spatio_x, n_grid_spatio_y]
    n_grid_time = int(T / delta_t) + 1

    L_s_x = int(2 * W_s[0] / delta_s[0]) + 1
    L_s_y = int(2 * W_s[1] / delta_s[1]) + 1
    L_s = [L_s_x, L_s_y]
    L_t = int(W_t / delta_t) + 1
    
    events_grid, events_smooth, coord_grid =\
        smooth_projection_spatio_temp(events, n_grid_spatio, n_grid_time, 
                                    delta_s, delta_t, spatio_bound)

    n_events = [len(events[j]) for j in range(1)]
    
    prepsi, zN = precomp_phi_events_prepsi(coord_grid, kernel_length, events_smooth,
                        L_s, L_t, delta, n_events)

    indices = np.indices((L_s[0], L_s[1], L_t, L_s[0], L_s[1], L_t), dtype=np.int8)
    A = prepsi[0][0]
    ztzG = A[np.abs(np.subtract(indices[0], indices[3]) + 2 * (L_s[0]//2)),
            np.abs(np.subtract(indices[1], indices[4]) + 2 * (L_s[1]//2)),
            np.abs(np.subtract(indices[2], indices[5]))]
    indices = 0

    time_prepsi = time.time() - start

    start = time.time()

    psi_loops = precomp_psi_loops(events_grid, L_s, L_t)
    psi_loops = psi_loops.numpy()

    time_truepsi = time.time() - start

    results_ = dict(time_prepsi=time_prepsi)

    M_tot = ztzG - psi_loops

    norm_frobenius_prepsi = np.sqrt(np.sum(ztzG**2))
    norm_1_prepsi = np.sum(np.abs(ztzG))
    norm_inf_prepsi = np.max(np.abs(ztzG))

    norm_frobenius_true = np.sqrt(np.sum(psi_loops**2))
    norm_1_true = np.sum(np.abs(psi_loops))
    norm_inf_true = np.max(np.abs(psi_loops))

    norm_frobenius = np.sqrt(np.sum(M_tot**2)) / norm_frobenius_true
    norm_1 = np.sum(np.abs(M_tot)) / norm_1_true
    norm_inf = np.max(np.abs(M_tot)) / norm_inf_true

    results_["time_prepsi"] = time_prepsi
    results_["time_truepsi"] = time_truepsi
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = delta[1]
    results_["K"] = K[0]
    results_["ds"] = delta[0][0]
    results_["normF"] = norm_frobenius
    results_["norm1"] = norm_1
    results_["norminf"] = norm_inf

    results_["normF_pre"] = norm_frobenius_prepsi
    results_["norm1_pre"] = norm_1_prepsi
    results_["norminf_pre"] = norm_inf_prepsi

    results_["normF_true"] = norm_frobenius_true
    results_["norm1_true"] = norm_1_true
    results_["norminf_true"] = norm_inf_true

    return results_

def run_experiment(baseline, alpha, kernel_params_time, kernel_params_space_x,\
                    kernel_params_space_y, kernel, T, K, spatio_bound, Wt, Ws, dt, ds, seed=0):
    v = 0.2

    events = simulate_data(T, K, Wt, Ws, baseline, alpha, kernel_params_time,\
            kernel_params_space_x, kernel_params_space_y, seed=seed)

    results = run_matrices(events, T, K, spatio_bound, [Ws, Wt], [ds, dt], seed=seed)

    return results

T_list = [5, 10, 50]
K_list = [5, 10]
dt_list = [0.5]
ds_list = [0.5]
seeds = np.arange(1)

n_jobs = 1
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline_true_fixed, alpha_true_fixed, kernel_params_time,\
                            kernel_params_space_x, kernel_params_space_y, kernel, T, [K, K], [-K, -K], Wt, Ws,\
                            dt, [ds, ds], seed=seed)
    for T, dt, K, ds, seed in itertools.product(
        T_list, dt_list, K_list, ds_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)

df.to_csv('results/exp4_PSI_APPROX_TG.csv', index=False)