import torch
import time
import matplotlib.pyplot as plt
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

from spatio_temporal_fadin.utils.discretisation import *
from spatio_temporal_fadin.utils.compute_constants import *
from spatio_temporal_fadin.loss_and_gradient import *
from spatio_temporal_fadin.kernel import *
from spatio_temporal_fadin.utils.functions import optimizer

class SpatialFaDIn(object):


    """Define the SpatialFaDIn framework for estimated Hawkes processes.

    The framework is detailed in:

    Emilia Siviero, Guillaume Staerman, Stephan Clemencon, Thomas Moreau
    Flexible Parametric Inference for Space-Time Hawkes Processes
    https://arxiv.org/abs/2406.06849


    Parameters
    ----------
    n_dim : `int`
        Dimension of the underlying Hawkes process.

    kernel : 'array' of `str`, shape (2,)
        Spatial (in {'truncated_gaussian' | 'inverse_power_law'}) \
        and temporal (in {'truncated_exponential' | 'truncated_gaussian' \
        | 'raised_cosine' | 'kumaraswamy'} kernels.

    kernel_params_init : `list` of tensor of shape (n_dim, n_dim)
        Initial parameters of the kernel.

    baseline_init : `tensor`, shape (n_dim,)
        Initial baseline parameters of the intensity of the Hawkes process.

    alpha_init : `tensor`, shape (n_dim, n_dim)
        Initial alpha parameters of the intensity of the Hawkes process.

    kernel_length: 'array', shape (2,)
        Spatial (of shape (2,)) and temporal kernel lengths.

    delta : 'array', shape (2,)
        Spatial (shape (2,)) and temporal step sizes of the discretization.

    optim : `str` in ``{'RMSprop' | 'Adam' | 'GD'}``, default='RMSprop'
        The algorithms used to optimize the Hawkes processes parameters.

    max_iter : `int`, `default=1000`
        Maximum number of iterations during fit.

    device : `str` in ``{'cpu' | 'cuda'}``
        Computations done on cpu or gpu. Gpu is not implemented yet.

    grad_kernel : 'array' of `str`, shape (2,)
        Spatial (in {'truncated_gaussian' | 'inverse_power_law'}) \
        and temporal (in {'truncated_exponential' | 'truncated_gaussian' \
        | 'raised_cosine' | 'kumaraswamy'} gradient functions.

    criterion : `str` in ``{'l2' | 'll'}``, `default='l2'`
        The criterion to minimize. if not l2, FaDIn minimize
        the Log-Likelihood loss through AutoDifferentiation.

    tol : `float`, `default=1e-5`
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). If not reached the solver does 'max_iter'
        iterations.

    random_state : `int`, `RandomState` instance or `None`, `default=None`
        Set the torch seed to 'random_state'.
    """

    def __init__(self, n_dim, kernel, kernel_params_init,
                 baseline_init, alpha_init,
                 kernel_length, delta, optim='RMSprop',
                 params_optim=dict(), max_iter=2000, device='cpu', grad_kernel=None,
                 criterion='l2', tol=10e-5, random_state=None):
        
        # Spatial param discretisation
        self.delta_s = delta[0]
        self.delta_s_x = self.delta_s[0]
        self.delta_s_y = self.delta_s[1]

        self.W_s = kernel_length[0]
        self.W_s_x = self.W_s[0]
        self.W_s_y = self.W_s[1]

        self.L_s_x = int(2 * self.W_s_x / self.delta_s_x) + 1
        self.L_s_y = int(2 * self.W_s_y / self.delta_s_y) + 1
        self.L_s = [self.L_s_x, self.L_s_y]

        # Temporal param discretisation
        self.delta_t = delta[1]
        self.W_t = kernel_length[1]
        self.L_t = int(self.W_t / self.delta_t) + 1

        self.n_dim = n_dim
        self.delta = [self.delta_s, self.delta_t]
        self.W = [self.W_s, self.W_t]

        # param optim
        self.solver = optim
        self.max_iter = max_iter
        self.tol = tol

        # Model Parameters
        # Baseline
        self.baseline = baseline_init.float().requires_grad_(True)
        
        # Alpha
        self.alpha = alpha_init.float().requires_grad_(True)

        # Kernel Parameters
        self.kernel_params_fixed = kernel_params_init

        self.n_kernel_params = len(kernel_params_init)

        # Kernel Model
        self.kernel_model = DiscreteKernelFiniteSupport(self.delta, self.n_dim,
                                                        kernel, self.W,
                                                        [[-self.W_s[0],-self.W_s[1]],0],
                                                        self.W, grad_kernel)
        self.kernel = kernel

        # Set optimizer
        self.params_intens = [self.baseline, self.alpha]

        for i in range(self.n_kernel_params):
            self.params_intens.append(
                kernel_params_init[i].float().clip(1e-3).requires_grad_(True))
        
        # If the learning rate is not given, fix it to 1e-3
        if 'lr' in params_optim.keys():
            params_optim['lr'] = 1e-3

        self.opt = optimizer(self.params_intens, params_optim, solver=optim)
        if criterion == 'll':
            self.precomputations = False

        self.criterion = criterion
        # device and seed
        if random_state is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(random_state)

        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def fit(self, events, end_time, spatio_bound, S):
        """Learn the parameters of the Hawkes processes on a discrete grid.

        Parameters
        ----------
        events : list of array of size number of timestamps, shape (n_dim,)

        end_time : `int`
            T the stopping time of the MSTH process

        spatio_bound : list of shape (2,)
            Spatial bounds
    
        S : list of shape (2,)
            [S_X, S_Y] the spatial domain

        Returns
        -------
        self : object
            Fitted parameters.
        """
        n_grid_spatio_x = int(2 * S[0] / self.delta_s[0]) + 1
        n_grid_spatio_y = int(2 * S[1] / self.delta_s[1]) + 1
        n_grid_spatio = [n_grid_spatio_x, n_grid_spatio_y]
        n_grid_time = int(end_time / self.delta_t) + 1

        discretization_spatio_support_x = torch.linspace(-self.W_s[0], self.W_s[0],
                                                         self.L_s[0])
        discretization_spatio_support_y = torch.linspace(-self.W_s[1], self.W_s[1],
                                                         self.L_s[1])
        
        discretization_time_support = torch.linspace(0, self.W_t, self.L_t)
        
        events_grid, events_smooth, coord_grid =\
            smooth_projection_spatio_temp(events, n_grid_spatio, n_grid_time, 
                                        self.delta_s, self.delta_t, spatio_bound)

        n_events = [len(events[j]) for j in range(self.n_dim)]

        ####################################################
        # Precomputations
        ####################################################
        
        zG = precomp_phi_grid(self.L_s, self.L_t, events_grid, n_events)
        zG = torch.tensor(zG).float()
        
        prepsi, zN = precomp_phi_events_prepsi(coord_grid, self.W, events_smooth,
                        self.L_s, self.L_t, n_events)
        zN = torch.tensor(zN).float()

        indices = np.indices((self.L_s[0], self.L_s[1], self.L_t, self.L_s[0],
                              self.L_s[1], self.L_t), dtype=np.int8)
        A = prepsi[0][0]
        ztzG = A[np.abs(np.subtract(indices[0], indices[3]) + 2 * (self.L_s[0]//2)),
                np.abs(np.subtract(indices[1], indices[4]) + 2 * (self.L_s[1]//2)),
                np.abs(np.subtract(indices[2], indices[5]))]
        indices = 0
        ztzG = torch.tensor(ztzG).float()

        zG = zG.to(device)
        zN = zN.to(device)
        ztzG = ztzG.to(device)

        n_events_tensor = torch.tensor(n_events).float()

        ####################################################
        # save results
        ####################################################

        self.param_baseline = torch.zeros(self.max_iter + 1, self.n_dim)
        self.param_baseline[0] = self.params_intens[0].detach()

        self.param_alpha = torch.zeros(self.max_iter + 1,
                                        self.n_dim,
                                        self.n_dim)
        self.param_alpha[0] = self.params_intens[1].detach()

        if self.kernel[0] == "truncated_gaussian"\
            or self.kernel[0] == "inverse_power_law":
            self.param_kernel_spatial_mean = torch.zeros(self.max_iter + 1,
                                            self.n_dim, self.n_dim, 2)
            self.param_kernel_spatial_sigma = torch.zeros(self.max_iter + 1,
                                            self.n_dim, self.n_dim, 2, 2)
            self.param_kernel_spatial_mean[0] = self.params_intens[2].detach()
            self.param_kernel_spatial_sigma[0] = self.params_intens[3].detach()
        else:
            self.param_kernel_spatial = torch.zeros(self.max_iter + 1,
                                            self.n_dim, self.n_dim)
            self.param_kernel_spatial[0] = self.params_intens[2].detach()

        if self.kernel[1] == "truncated_gaussian" or self.kernel[1] == 'kumaraswamy'\
              or self.kernel[1] == 'raised_cosine':
            self.param_kernel_temporal_mean = torch.zeros(self.max_iter + 1,
                                            self.n_dim, self.n_dim, 2)
            self.param_kernel_temporal_sigma = torch.zeros(self.max_iter + 1,
                                            self.n_dim, self.n_dim, 2, 2)
            if self.kernel[0] == "truncated_gaussian"\
                  or self.kernel[0] == "inverse_power_law":
                self.param_kernel_temporal_mean[0] = self.params_intens[4].detach()
                self.param_kernel_temporal_sigma[0] = self.params_intens[5].detach()
            else:
                self.param_kernel_temporal_mean[0] = self.params_intens[3].detach()
                self.param_kernel_temporal_sigma[0] = self.params_intens[4].detach()
        else:
            self.param_kernel_temporal = torch.zeros(self.max_iter + 1,
                                            self.n_dim, self.n_dim)
            if self.kernel[0] == "truncated_gaussian"\
                  or self.kernel[0] == "inverse_power_law":
                self.param_kernel_temporal[0] = self.params_intens[4].detach()
            else:
                self.param_kernel_temporal[0] = self.params_intens[3].detach()
        ####################################################
        start_iter = time.time()
        for i in range(self.max_iter):
            # print(f"Fitting model... {i/self.max_iter:6.1%}\r", end='',
            #         flush=True)

            self.opt.zero_grad()
            
            # Update kernel
            kernel = self.kernel_model.kernel_eval(self.params_intens[2:],
                                                discretization_spatio_support_x,
                                                discretization_spatio_support_y,
                                                discretization_time_support)
            kernel_de = kernel.to(device)
            
            grad_kernel = self.kernel_model.grad_eval(self.params_intens[2:],
                                                discretization_spatio_support_x,
                                                discretization_spatio_support_y,
                                                discretization_time_support)

            # Update baseline
            self.params_intens[0].grad = get_grad_baseline(zG,
                                                            self.params_intens[0],
                                                            self.params_intens[1],
                                                            kernel_de, self.delta,
                                                            n_events_tensor,
                                                            end_time, S)

            # Update alpha
            self.params_intens[1].grad = get_grad_alpha(zG, ztzG, zN,
                                                        self.params_intens[0],
                                                        self.params_intens[1],
                                                        kernel_de, self.delta,
                                                        n_events_tensor)

            # Update kernel parameters
            if self.kernel[0] == "truncated_gaussian"\
                  or self.kernel[0] == "inverse_power_law":
                grad_kernel_mean = (grad_kernel[0]).to(device)
                grad_kernel_sigma = (grad_kernel[1]).to(device)
                grad_mean, grad_sigma =\
                    get_grad_eta_gaussian(zG, ztzG, zN, self.params_intens[0],
                                          self.params_intens[1], kernel_de,
                                          grad_kernel_mean, grad_kernel_sigma,
                                          self.delta, n_events_tensor)
                self.params_intens[2].grad = grad_mean
                self.params_intens[3].grad = grad_sigma

                if self.kernel[1] == "truncated_gaussian"\
                      or self.kernel[1] == 'kumaraswamy'\
                          or self.kernel[1] == 'raised_cosine':
                    grad_kernel_mean_time = (grad_kernel[2]).to(device)
                    grad_kernel_sigma_time = (grad_kernel[3]).to(device)
                    grad_mean_time, grad_sigma_time =\
                        get_grad_eta_gaussian(zG, ztzG, zN, self.params_intens[0],
                                              self.params_intens[1], kernel_de,
                                              grad_kernel_mean_time,
                                              grad_kernel_sigma_time,
                                              self.delta, n_events_tensor)
                    self.params_intens[4].grad = grad_mean_time
                    self.params_intens[5].grad = grad_sigma_time

                else:
                    grad_kernel_time = (grad_kernel[2]).to(device)
                    self.params_intens[4].grad =\
                        get_grad_eta(zG, ztzG, zN, self.params_intens[0],
                                     self.params_intens[1], kernel_de, grad_kernel_time,
                                     self.delta, n_events_tensor)
            else:
                grad_kernel_spatial = (grad_kernel[0]).to(device)
                self.params_intens[2].grad =\
                    get_grad_eta(zG, ztzG, zN,self.params_intens[0],
                                 self.params_intens[1], kernel_de, grad_kernel_spatial,
                                 self.delta, n_events_tensor)
                
                if self.kernel[1] == "truncated_gaussian"\
                    or self.kernel[1] == 'kumaraswamy'\
                        or self.kernel[1] == 'raised_cosine':
                    grad_kernel_mean_time = (grad_kernel[1]).to(device)
                    grad_kernel_sigma_time = (grad_kernel[2]).to(device)
                    grad_mean_time, grad_sigma_time =\
                        get_grad_eta_gaussian(zG, ztzG, zN, self.params_intens[0],
                                              self.params_intens[1], kernel_de,
                                              grad_kernel_mean_time,
                                              grad_kernel_sigma_time,
                                              self.delta, n_events_tensor)
                    self.params_intens[3].grad = grad_mean_time
                    self.params_intens[4].grad = grad_sigma_time

                else:
                    grad_kernel_time = (grad_kernel[2]).to(device)
                    self.params_intens[3].grad =\
                        get_grad_eta(zG, ztzG, zN, self.params_intens[0],
                                     self.params_intens[1], kernel_de, grad_kernel_time,
                                     self.delta, n_events_tensor)
            
            

            self.opt.step()
            # Save parameters
            self.param_baseline[i + 1] = self.params_intens[0].detach()
            self.params_intens[0].data = self.params_intens[0].clamp(min=1e-3)
            
            self.param_alpha[i + 1] = self.params_intens[1].detach()
            self.params_intens[1].data = self.params_intens[1].clamp(min=1e-3)

            for j in range(self.n_kernel_params):
                self.params_intens[2 + j].data = \
                    self.params_intens[2 + j].clamp(min=1e-3)
                
            if self.kernel[0] == "truncated_gaussian"\
                or self.kernel[0] == "inverse_power_law":
                self.param_kernel_spatial_mean[i + 1] = self.params_intens[2].detach()
                self.param_kernel_spatial_sigma[i + 1] = self.params_intens[3].detach()
                
                if self.kernel[1] == "truncated_gaussian"\
                    or self.kernel[1] == 'kumaraswamy'\
                        or self.kernel[1] == 'raised_cosine':
                    self.param_kernel_temporal_mean[i + 1] =\
                        self.params_intens[4].detach()
                    self.param_kernel_temporal_sigma[i + 1] =\
                        self.params_intens[5].detach()
                else:
                    self.param_kernel_temporal[i + 1] = self.params_intens[4].detach()

            else:
                self.param_kernel_spatial[i + 1] = self.params_intens[2].detach()

                if self.kernel[1] == "truncated_gaussian"\
                    or self.kernel[1] == 'kumaraswamy'\
                        or self.kernel[1] == 'raised_cosine':
                    self.param_kernel_temporal_mean[i + 1] =\
                        self.params_intens[3].detach()
                    self.param_kernel_temporal_sigma[i + 1] =\
                        self.params_intens[4].detach()
                else:
                    self.param_kernel_temporal[i + 1] = self.params_intens[3].detach()


            # Early stopping
            if i % 100 == 0:
                error_b = torch.abs(self.param_baseline[i + 1] -
                                    self.param_baseline[i]).max()
                error_al = torch.abs(self.param_alpha[i + 1] -
                                        self.param_alpha[i]).max()
                if self.kernel[0] == "truncated_gaussian"\
                    or self.kernel[0] == "inverse_power_law":
                    error_mean = torch.abs(self.param_kernel_spatial_mean[i + 1] -
                                        self.param_kernel_spatial_mean[i]).max()
                    error_sigma = torch.abs(self.param_kernel_spatial_sigma[i + 1] -
                                        self.param_kernel_spatial_sigma[i]).max()
                    error_spatial = 0
                else:
                    error_spatial = torch.abs(self.param_kernel_spatial[i + 1] -
                                        self.param_kernel_spatial[i]).max()
                    error_mean = 0
                    error_sigma = 0
                if self.kernel[1] == "truncated_gaussian"\
                    or self.kernel[1] == 'kumaraswamy'\
                        or self.kernel[1] == 'raised_cosine':
                    error_mean_time = torch.abs(self.param_kernel_temporal_mean[i+1] -
                                        self.param_kernel_temporal_mean[i]).max()
                    error_sigma_time = torch.abs(self.param_kernel_temporal_sigma[i+1] -
                                        self.param_kernel_temporal_sigma[i]).max()
                    error_time = 0
                else:
                    error_time = torch.abs(self.param_kernel_temporal[i + 1] -
                                        self.param_kernel_temporal[i]).max()
                    error_mean_time = 0
                    error_sigma_time = 0

                if error_b < self.tol and error_al < self.tol \
                    and error_mean < self.tol and error_sigma < self.tol \
                        and error_spatial < self.tol and error_time < self.tol \
                            and error_mean_time < self.tol \
                                and error_sigma_time < self.tol:
                    print('early stopping at iteration:', i)
                    self.iteration_max = i
                    self.param_baseline = self.param_baseline[:i + 1]
                    self.param_alpha = self.param_alpha[:i + 1]

                    if self.kernel[0] == "truncated_gaussian"\
                        or self.kernel[0] == "inverse_power_law":
                        self.param_kernel_spatial_mean =\
                            self.param_kernel_spatial_mean[:i+1]
                        self.param_kernel_spatial_sigma =\
                            self.param_kernel_spatial_sigma[:i+1]
                    else:
                        self.param_kernel_spatial = self.param_kernel_spatial[:i+1]

                    if self.kernel[1] == "truncated_gaussian"\
                        or self.kernel[1] == 'kumaraswamy'\
                            or self.kernel[1] == 'raised_cosine':
                        self.param_kernel_temporal_mean =\
                            self.param_kernel_temporal_mean[:i+1]
                        self.param_kernel_temporal_sigma =\
                            self.param_kernel_temporal_sigma[:i+1]
                    else:
                        self.param_kernel_temporal = self.param_kernel_temporal[:i+1]
                    break

        print('Iterations in ', time.time() - start_iter)

        self.iteration_max = i

        return self