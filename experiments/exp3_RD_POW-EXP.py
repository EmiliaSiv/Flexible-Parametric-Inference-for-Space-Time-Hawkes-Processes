# %% import stuff
# import libraries

import itertools
import pandas as pd
import time
import numpy as np
import torch
from joblib import Memory, Parallel, delayed
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from spatio_temporal_fadin.utils.simulation import *
from spatio_temporal_fadin.kernel import DiscreteKernelFiniteSupport
from spatio_temporal_fadin.solver import SpatialFaDIn                          
from spatio_temporal_fadin.loss_and_gradient import *
from spatio_temporal_fadin.utils.discretisation import smooth_projection_spatio_temp


################################
# Meta parameters
################################


mem = Memory(location=".", verbose=0)

@mem.cache
def run_solver(events, T, K, spatio_bound, kernel, kernel_params_tensor_init, baseline_tensor_init,\
                alpha_tensor_init, kernel_length, delta, seed=0):

    max_iter = 2000
    start = time.time()
    solv = SpatialFaDIn(1, kernel, kernel_params_init=kernel_params_tensor_init,
                 baseline_init=baseline_tensor_init, alpha_init=alpha_tensor_init,
                 kernel_length=kernel_length, delta=delta, optim='RMSprop',
                 params_optim=dict(), max_iter=max_iter, device='cpu',
                 grad_kernel=kernel, criterion='l2', tol=10e-5, random_state=seed)

    results = solv.fit(events, T, spatio_bound, K)

    results_ = dict(param_baseline=solv.param_baseline[-10:].mean().item(),
                    param_alpha=solv.param_alpha[-10:].mean().item(),
                    param_kernel_spatial=[solv.param_kernel_spatial_mean[-10:].mean().item(),
                                  solv.param_kernel_spatial_sigma[-10:].mean().item()],
                    param_kernel_temporal=solv.param_kernel_temporal[-10:].mean().item())

    results_["time"] = time.time() - start
    results_["seed"] = seed
    results_["T"] = T
    results_["dt"] = delta[1]
    results_["K"] = K
    results_["ds"] = delta[0]
    return results_

def run_experiment(events, kernel, T, K, spatio_bound, Wt, Ws, dt, ds, seed=0):

    train_events, test_events = train_test_split(events[0], test_size=0.2, random_state=seed)
    train_events = train_events[train_events[:, 2].argsort()]
    test_events = test_events[test_events[:, 2].argsort()]

    baseline_init = torch.tensor([[0.3]])
    alpha_init = torch.tensor([[0.5]])
    mean_init = torch.tensor(([[0.5]]))
    sigma_init = torch.tensor(([[0.3]]))
    decay_init = torch.tensor(([[0.5]]))
    kernel_params_init = [mean_init, sigma_init, decay_init]

    results = run_solver([train_events], T, K, spatio_bound, kernel, kernel_params_init, baseline_init,
                         alpha_init, [Ws, Wt], [ds, dt], seed=seed)

    baseline_train = torch.tensor([[results["param_baseline"]]])
    alpha_train = torch.tensor([[results["param_alpha"]]])
    mean_train = torch.tensor(([[results['param_kernel_spatial'][0]]]))
    sigma_train = torch.tensor(([[results['param_kernel_spatial'][1]]]))
    decay_train = torch.tensor(([[results['param_kernel_temporal']]]))

    kernel_params_train = [mean_train, sigma_train, decay_train]

    delta = [ds, dt]
    Ls_x = int(2 * Ws[0] / ds[0]) + 1
    Ls_y = int(2 * Ws[1] / ds[1]) + 1
    Lt = int(Wt / dt) + 1
    n_grid_spatio_x = int(2 * K[0] / ds[0]) + 1
    n_grid_spatio_y = int(2 * K[1] / ds[1]) + 1
    n_grid_spatio = [n_grid_spatio_x, n_grid_spatio_y]
    n_grid_time = int(T / dt) + 1
    
    discretization_time_support = torch.linspace(0, Wt, Lt)
    discretization_spatio_support_x = torch.linspace(-Ws[0], Ws[0], Ls_x)
    discretization_spatio_support_y = torch.linspace(-Ws[1], Ws[1], Ls_y)

    events_grid_test, _, _ = smooth_projection_spatio_temp([test_events], n_grid_spatio,
                            n_grid_time, ds, dt, spatio_bound)

    kernel_model = DiscreteKernelFiniteSupport([ds, dt], n_dim, kernel, [Ws, Wt],
                                            [[-Ws[0], -Ws[1]], Wt], [Ws, Wt], kernel)
    intensity_values = kernel_model.intensity_eval(baseline_train, alpha_train, kernel_params_train,
                       torch.tensor(events_grid_test).float(), discretization_spatio_support_x,
                                            discretization_spatio_support_y,
                                            discretization_time_support)

    NLL = negative_log_likelihood(intensity_values, events_grid_test, delta)

    results["NLL"] = NLL

    return results


T = 5

from spatio_temporal_fadin.utils.real_data import *

n_dim, events, spatio_bound, K = read_real_data("data_NCEDC/real_data_NCEDC_1978-2018", T)

kernel = ["inverse_power_law", "truncated_exponential"]

T_list = [T]
dt_list = [0.05]
ds_list = [0.05]
W_s_list = [1]
W_t_list = [1]
seeds = np.arange(1)

n_jobs = 1
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(events, kernel, T, K, spatio_bound, Wt, [Ws, Ws],\
                            dt, [ds, ds], seed=seed)
    for T, dt, ds, Wt, Ws, seed in itertools.product(
        T_list, dt_list, ds_list, W_t_list, W_s_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)

df['param_mean'] = df['param_kernel_spatial'].apply(lambda x: x[0])
df['param_sigma'] = df['param_kernel_spatial'].apply(lambda x: x[1])
df['param_decay'] = df['param_kernel_temporal']

df.to_csv('results/exp3_RD_POW-EXP.csv', index=False)