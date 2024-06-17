# %% import stuff
# import libraries

import itertools
import pandas as pd
import time
import numpy as np
import torch
from joblib import Memory, Parallel, delayed

from spatio_temporal_fadin.utils.simulation import *
from spatio_temporal_fadin.kernel import DiscreteKernelFiniteSupport
from spatio_temporal_fadin.solver import SpatialFaDIn


################################
# Meta parameters
################################

spatio_bound_x = 10
spatio_bound_y = 10
K = [spatio_bound_x, spatio_bound_y]
spatio_bound = [-spatio_bound_x, -spatio_bound_y]
delta_spatio_x = 0.5
delta_spatio_y = 0.5
delta_spatio = [delta_spatio_x, delta_spatio_y]
Ws_x = 1
Ws_y = 1
Ws = [Ws_x, Ws_y]

delta_time = 0.5
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
def run_solver(events, T, K, spatio_bound, kernel, kernel_params_tensor_init, baseline_tensor_init,\
                alpha_tensor_init, kernel_length, delta, seed=0):

    max_iter = 2000
    start = time.time()
    solv = SpatialFaDIn(1, kernel, kernel_params_init=kernel_params_tensor_init,
                 baseline_init=baseline_tensor_init, alpha_init=alpha_tensor_init,
                 kernel_length=kernel_length, delta=delta, optim='RMSprop',
                 params_optim=dict(), max_iter=max_iter, device='cpu',
                 grad_kernel=grad_kernel, criterion='l2', tol=10e-5, random_state=seed)

    results = solv.fit(events, T, spatio_bound, K)

    results_ = dict(param_baseline=solv.param_baseline[-10:].mean().item(),
                    param_alpha=solv.param_alpha[-10:].mean().item(),
                    param_kernel_spatial=[solv.param_kernel_spatial_mean[-10:].mean().item(),
                                  solv.param_kernel_spatial_sigma[-10:].mean().item()],
                    param_kernel_temporal=[solv.param_kernel_temporal_mean[-10:].mean().item(),
                                  solv.param_kernel_temporal_sigma[-10:].mean().item()])

    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = delta[1]
    results_["K"] = K[0]
    results_["ds"] = delta[0][0]
    return results_

def run_experiment(baseline, alpha, kernel_params_time, kernel_params_space_x,\
                    kernel_params_space_y, kernel, T, K, spatio_bound, Wt, Ws, dt, ds, seed=0):
    v = 0.2

    events = simulate_data(T, K, Wt, Ws, baseline, alpha, kernel_params_time,\
            kernel_params_space_x, kernel_params_space_y, seed=seed)

    mu = baseline[0]
    alp = alpha[0][0]
    
    baseline_init = torch.tensor([[mu + v]])
    alpha_init = torch.tensor([[alp + v]])
    mean_init = kernel_params_space_x[0] + v
    sigma_init = kernel_params_space_x[1] + v
    mean_time_init = kernel_params_time[0] + v
    sigma_time_init = kernel_params_time[1] + v
    kernel_params_init = [mean_init, sigma_init, mean_time_init, sigma_time_init]

    results = run_solver(events, T, K, spatio_bound, kernel, kernel_params_init, baseline_init,
                         alpha_init, [Ws, Wt], [ds, dt], seed=seed)

    return results

T_list = [10, 10**2, 10**3, 10**4]
K_list = [10, 20]
seeds = np.arange(100)

n_jobs = 10
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline_true_fixed, alpha_true_fixed, kernel_params_time,\
                            kernel_params_space_x, kernel_params_space_y, kernel, T, [K, K], [-K, -K], Wt, Ws,\
                            delta_time, [delta_spatio_x, delta_spatio_y], seed=seed)
    for T, K, seed in itertools.product(
        T_list, K_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)

df['param_mean'] = df['param_kernel_spatial'].apply(lambda x: x[0])
df['param_sigma'] = df['param_kernel_spatial'].apply(lambda x: x[1])
df['param_mean_time'] = df['param_kernel_temporal'].apply(lambda x: x[0])
df['param_sigma_time'] = df['param_kernel_temporal'].apply(lambda x: x[1])
true_param = {'baseline': mu, 'alpha': alpha, 'mean': mean, 'sigma': sigma, 'mean_time': mean_time, 'sigma_time': sigma_time}
for param, value in true_param.items():
    df[param] = value


def compute_norm2_error(s):
    return np.sqrt(np.array([(s[param] - s[f'param_{param}'])**2
                            for param in ['baseline', 'alpha', 'mean', 'sigma', 'mean_time', 'sigma_time']]).sum())


df['err_norm2'] = df.apply(
    lambda x: compute_norm2_error(x), axis=1)

df.to_csv('results/exp2_TG.csv', index=False)