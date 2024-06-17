import torch
import numpy as np

import torch.nn.functional as F

from spatio_temporal_fadin.utils.functions import *

class DiscreteKernelFiniteSupport(object):
    """
    A class for general discretized kernels with finite support.

    Parameters
    ----------
    grid_step : 'array', shape (2,)
        Spatial (shape (2,)) and temporal step sizes of the discretization.

    n_dim : `int`
        Dimension of the Hawkes process associated to this kernel class.

    kernel : 'array' of `str`, shape (2,)
        Spatial (in {'truncated_gaussian' | 'inverse_power_law'}) \
        and temporal (in {'truncated_exponential' | 'truncated_gaussian' \
        | 'raised_cosine' | 'kumaraswamy'} kernels.

    kernel_length: 'array', shape (2,)
        Spatial (of shape (2,)) and temporal kernel lengths.

    lower : `array` of float, shape (2,)
        Left bound of the support of the kernel.

    upper : `array` of float, shape (2,)
        Right bound of the support of the kernel.

    grad_kernel : 'array' of `str`, shape (2,)
        Spatial (in {'truncated_gaussian' | 'inverse_power_law'}) \
        and temporal (in {'truncated_exponential' | 'truncated_gaussian' \
        | 'raised_cosine' | 'kumaraswamy'} gradient functions.
    """
    def __init__(self, grid_step, n_dim, kernel, kernel_length,
                 lower, upper, grad_kernel):
        self.n_dim = n_dim

        # Spatial
        kernel_length_spatial = kernel_length[0]
        # lower bound
        self.lower_s = lower[0]
        self.lower_s_x = self.lower_s[0]
        self.lower_s_y = self.lower_s[1]
        # upper bound
        self.upper_s = kernel_length[0]
        self.upper_s_x = self.upper_s[0]
        self.upper_s_y = self.upper_s[1]
        # delta_spatio
        self.delta_s = grid_step[0]
        self.delta_s_x = self.delta_s[0]
        self.delta_s_y = self.delta_s[1]
        # n_support_spatio
        self.L_s_x = int(2 *kernel_length_spatial[0] / self.delta_s_x)+ 1
        self.L_s_y = int(2 *kernel_length_spatial[1] / self.delta_s_y)+ 1
        self.L_s = [self.L_s_x, self.L_s_y]

        self.kernel_s = kernel[0]
        self.grad_kernel_s = grad_kernel[0]

        # Temporal
        # n_support_time
        self.L_t = int(kernel_length[1] / grid_step[1]) + 1
        # delta_time
        self.delta_t = grid_step[1]
        # lower bound
        self.lower_t = lower[1]
        # upper bound
        self.upper_t = kernel_length[1]
        self.kernel_t = kernel[1]
        self.grad_kernel_t = grad_kernel[1]

        # Spatial
        if self.upper_s_x < self.lower_s_x or self.upper_s_y < self.lower_s_y:
            raise AttributeError('Upper bound must be higher than the lower bound \
                (for Spatial)')

        if self.upper_s_x > kernel_length[0][0] or self.upper_s_y > kernel_length[0][1]:
            raise AttributeError('Upper bound must be lower than the kernel length \
                (for Spatial)')
        
        # Temporal
        if self.upper_t < self.lower_t:
            raise AttributeError('Upper bound must be higher than the lower bound \
                (for Temporal)')

        if self.upper_t > kernel_length[1]:
            raise AttributeError('Upper bound must be lower than the kernel length \
                (for Temporal)')

        if self.lower_t < 0:
            raise AttributeError('Lower bound must be higher than zero \
                (for Temporal)')

    def kernel_eval(self, kernel_params, space_values_x, space_values_y, time_values):
        """Return kernel evaluated on the given discretization.

        Parameters
        ----------
        kernel_params : `list` of tensor of shape (n_dim, n_dim, 2)
            Parameters of the kernel.

        space_values_x : `tensor`, shape (L_X,)
            Given discretization.

        space_values_y : `tensor`, shape (L_Y,)
            Given discretization.

        time_values : `tensor`, shape (L_T,)
            Given discretization.

        Returns
        -------
        kernel_values :  `tensor`, shape (n_dim, n_dim, L_X, L_Y, L_T)
            Kernels evaluated on the discretized grid.
        """

        if self.kernel_s == 'truncated_gaussian' or self.kernel_s=='inverse_power_law':
            kernel_params_s = [kernel_params[0], kernel_params[1]]
            if self.kernel_t == 'truncated_gaussian' or self.kernel_t == 'kumaraswamy' \
                or self.kernel_t == 'raised_cosine':
                kernel_params_t = [kernel_params[2], kernel_params[3]]
            else:
                kernel_params_t = [kernel_params[2]]
        else:
            kernel_params_s = [kernel_params[0]]
            if self.kernel_t == 'truncated_gaussian' or self.kernel_t == 'kumaraswamy' \
                or self.kernel_t == 'raised_cosine':
                kernel_params_t = [kernel_params[1], kernel_params[2]]
            else:
                kernel_params_t = [kernel_params[1]]

        if self.kernel_s == 'truncated_gaussian':
            kernel_values_s = truncated_gaussian_spatial(kernel_params_s,
                                                 space_values_x, space_values_y,
                                                 self.delta_s,
                                                 self.lower_s,
                                                 self.upper_s)
        elif self.kernel_s == 'inverse_power_law':
            kernel_values_s = truncated_inv_power_law_spatial(kernel_params_s,
                                                 space_values_x, space_values_y,
                                                 self.delta_s,
                                                 self.lower_s,
                                                 self.upper_s)
        else:
            raise NotImplementedError("Not implemented kernel. \
                                       Kernel must be truncated_gaussian\
                                      or inverse_power_law")

        if self.kernel_t == 'truncated_exponential':
            kernel_values_t = truncated_exponential_temporal(kernel_params_t, time_values,
                                                  self.delta_t, self.upper_t)
        elif self.kernel_t == 'truncated_gaussian':
            kernel_values_t = truncated_gaussian_temporal(kernel_params_t, time_values,
                                                  self.delta_t, self.upper_t)
        elif self.kernel_t == 'kumaraswamy':
            kernel_values_t = kumaraswamy_temporal(kernel_params_t, time_values)
        elif self.kernel_t == 'raised_cosine':
            kernel_values_t = raised_cosine_temporal(kernel_params_t, time_values)
        else:
            raise NotImplementedError("Not implemented kernel. \
                                       Kernel must be truncated_exponential, truncated_gaussian, \
                                      kumaraswamy or raised_cosine")

        kernel_values = kernel_values_s[..., None] * kernel_values_t[..., None, None, :]

        kernel_values[:, :, self.L_s_x//2, self.L_s_y//2, 0] = 0
        
        return kernel_values

    def grad_eval(self, kernel_params, space_values_x, space_values_y, time_values):
        """Return spatial kernel's gradient evaluated on the given discretization.

        Parameters
        ----------
        kernel_params : `list` of tensor of shape (n_dim, n_dim, 2)
            Parameters of the spatial kernel.

        space_values_x : `tensor`, shape (L_X,)
            Given discretization.

        space_values_y : `tensor`, shape (L_Y,)
            Given discretization.

        time_values : `tensor`, shape (L_T,)
            Given discretization.

        Returns
        ----------
        grad_values :  `tensor`, shape (n_dim, n_dim, L_X, L_Y, L_T)
            Gradients evaluated on the discretized grid.
        """
        
        if self.kernel_s == 'truncated_gaussian' or self.kernel_s=='inverse_power_law':
            kernel_params_s = [kernel_params[0], kernel_params[1]]
            if self.kernel_t == 'truncated_gaussian' or self.kernel_t == 'kumaraswamy' \
                or self.kernel_t == 'raised_cosine':
                kernel_params_t = [kernel_params[2], kernel_params[3]]
            else:
                kernel_params_t = [kernel_params[2]]
        else:
            kernel_params_s = [kernel_params[0]]
            if self.kernel_t == 'truncated_gaussian' or self.kernel_t == 'kumaraswamy' \
                or self.kernel_t == 'raised_cosine':
                kernel_params_t = [kernel_params[1], kernel_params[2]]
            else:
                kernel_params_t = [kernel_params[1]]
            
        if self.kernel_s == 'truncated_gaussian':
            grad_values_s = grad_truncated_gaussian_spatial(kernel_params_s, 
                                                            space_values_x,
                                                            space_values_y,
                                                            self.delta_s, self.L_s)
            kernel_values_s = truncated_gaussian_spatial(kernel_params_s,
                                                space_values_x, space_values_y,
                                                 self.delta_s,
                                                 self.lower_s,
                                                 self.upper_s)
        elif self.kernel_s == 'inverse_power_law':
            grad_values_s = grad_truncated_inv_power_law_spatial(kernel_params_s,
                                                                 space_values_x,
                                                                 space_values_y,
                                                                 self.delta_s, self.L_s)
            kernel_values_s = truncated_inv_power_law_spatial(kernel_params_s,
                                                 space_values_x, space_values_y,
                                                 self.delta_s,
                                                 self.lower_s,
                                                 self.upper_s)
        else:
            raise NotImplementedError("Not implemented kernel. \
                                       Kernel and grad_kernel must be \
                                        truncated_gaussian or inverse_power_law")
        
        if self.kernel_t == 'truncated_exponential':
            grad_values_t = grad_truncated_exponential_temporal(kernel_params_t,
                                                                time_values,
                                                                self.delta_t)
            kernel_values_t = truncated_exponential_temporal(kernel_params_t,
                                                             time_values,
                                                             self.delta_t, self.upper_t)
        elif self.kernel_t == 'truncated_gaussian':
            grad_values_t = grad_truncated_gaussian_temporal(kernel_params_t,
                                                             time_values,
                                                             self.delta_t, self.L_t)
            kernel_values_t = truncated_gaussian_temporal(kernel_params_t, time_values,
                                                          self.delta_t, self.upper_t)
        elif self.kernel_t == 'kumaraswamy':
            grad_values_t = grad_kumaraswamy_temporal(kernel_params_t, time_values,
                                                      self.L_t)
            kernel_values_t = kumaraswamy_temporal(kernel_params_t, time_values)
        elif self.kernel_t == 'raised_cosine':
            grad_values_t = grad_raised_cosine_temporal(kernel_params_t, time_values,
                                                        self.L_t)
            kernel_values_t = raised_cosine_temporal(kernel_params_t, time_values)
        else:
            raise NotImplementedError("Not implemented kernel. \
                                       Kernel and grad_kernel must be \
                                        truncated_exponential, truncated_gaussian, \
                                      kumaraswamy or raised_cosine")
        
        if self.kernel_s == 'truncated_gaussian' or self.kernel_s=='inverse_power_law':
            grad_values_s_mean = grad_values_s[0]
            grad_values_s_sigma = grad_values_s[1]
            grad_values_spatial_mean = grad_values_s_mean[..., None]\
                 * kernel_values_t[..., None, None, :]
            grad_values_spatial_sigma = grad_values_s_sigma[..., None]\
                 * kernel_values_t[..., None, None, :]

            if self.kernel_t == 'truncated_gaussian' or self.kernel_t == 'kumaraswamy' \
                or self.kernel_t == 'raised_cosine':
                grad_values_t_mean = grad_values_t[0]
                grad_values_t_sigma = grad_values_t[1]
                grad_values_time_mean = kernel_values_s[..., None]\
                    * grad_values_t_mean[..., None, None, :]
                grad_values_time_sigma = kernel_values_s[..., None]\
                    * grad_values_t_sigma[..., None, None, :]

                return grad_values_spatial_mean, grad_values_spatial_sigma,\
                    grad_values_time_mean, grad_values_time_sigma
            
            else:
                grad_values_time = kernel_values_s[..., None]\
                      * grad_values_t[..., None, None, :]

                return grad_values_spatial_mean, grad_values_spatial_sigma,\
                    grad_values_time
        
        else:
            grad_values_spatial = grad_values_s[..., None]\
                  * kernel_values_t[..., None, None, :]
            
            if self.kernel_t == 'truncated_gaussian' or self.kernel_t == 'kumaraswamy'\
                or self.kernel_t == 'raised_cosine':
                grad_values_t_mean = grad_values_t[0]
                grad_values_t_sigma = grad_values_t[1]
                grad_values_time_mean = kernel_values_s[..., None]\
                      * grad_values_t_mean[..., None, None, :]
                grad_values_time_sigma = kernel_values_s[..., None]\
                      * grad_values_t_sigma[..., None, None, :]

                return grad_values_spatial, grad_values_time_mean,\
                    grad_values_time_sigma
            
            else:
                grad_values_time = kernel_values_s[..., None]\
                      * grad_values_t[..., None, None, :]

                return grad_values_spatial, grad_values_time

    def intensity_eval(self, baseline, alpha, kernel_params,
                       events_grid, space_values_x, space_values_y, time_values):
        """Return an approximation of the intensity function evaluated on the grid.

        Parameters
        ----------
        baseline : `tensor`, shape (n_dim,)
            Baseline parameter of the intensity of the Hawkes process.

        alpha : `tensor`, shape (n_dim, n_dim)
            Alpha parameter of the intensity of the Hawkes process.

        kernel_params : `list` of tensor of shape (n_dim, n_dim, 2)
            Parameters of the kernel.

        events_grid : `tensor`, shape (n_dim, n_grid_spatio[0], n_grid_spatio[1], \
        n_grid_time)
            Events projected on the pre-defined grid.

        space_values_x : `tensor`, shape (L_X,)
            Given discretization.

        space_values_y : `tensor`, shape (L_Y,)
            Given discretization.

        time_values : `tensor`, shape (L_T,)
            Given discretization.

        Returns
        ----------
        intensity_values : `tensor`, shape (n_dim, n_grid_spatio[0], n_grid_spatio[1], \
            n_grid_time)
            The intensity function evaluated on the grid.
        """
        kernel_values = self.kernel_eval(kernel_params, space_values_x, space_values_y,
                                         time_values)
        n_dim, G_s_x, G_s_y, G_T = events_grid.shape
        kernel_values_alp = kernel_values * alpha[:, :, None, None, None]

        intens_temp = torch.zeros(self.n_dim, self.n_dim, G_s_x, G_s_y, G_T)

        for i in range(self.n_dim):
            intens_temp[i, :, :, :, :] = torch.conv_transpose3d(
                events_grid[i].view(1, G_s_x, G_s_y, G_T), 
                kernel_values_alp[:, i].view(1, self.n_dim, self.L_s_x, self.L_s_y,
                                             self.L_t))[:, :-self.L_s_x+1, \
                                                        :-self.L_s_y+1, :-self.L_t+1]
            
        intensity_values = intens_temp.sum(0) + baseline.view(self.n_dim, 1, 1, 1)

        return intensity_values

    def intensity_eval_loops(self, baseline, alpha, kernel_params, events_grid,
                             space_values_x, space_values_y, time_values):
        """Return the intensity function evaluated on the grid.

        Parameters
        ----------
        baseline : `tensor`, shape (n_dim,)
            Baseline parameter of the intensity of the Hawkes process.

        alpha : `tensor`, shape (n_dim, n_dim)
            Alpha parameter of the intensity of the Hawkes process.

        kernel_params : `list` of tensor of shape (n_dim, n_dim, 2)
            Parameters of the kernel.

        events_grid : `tensor`, shape (n_dim, n_grid_spatio[0], n_grid_spatio[1], \
        n_grid_time)
            Events projected on the pre-defined grid.

        space_values_x : `tensor`, shape (L_X,)
            Given discretization.

        space_values_y : `tensor`, shape (L_Y,)
            Given discretization.

        time_values : `tensor`, shape (L_T,)
            Given discretization.

        Returns
        ----------
        intensity_values : `tensor`, shape (n_dim, n_grid_spatio[0], n_grid_spatio[1], \
            n_grid_time)
            The intensity function evaluated on the grid.
        """
        kernel_values = self.kernel_eval(kernel_params, space_values_x, space_values_y,
                                         time_values)
        n_dim, G_S_x, G_S_y, G_T = events_grid.shape

        L_x = self.L_s_x//2
        L_y = self.L_s_y//2

        intensity_values = torch.zeros(self.n_dim, G_S_x, G_S_y, G_T)

        for vx in range(G_S_x):
            for vy in range(G_S_y):
                for vt in range(G_T):
                    intensity_values[0, vx, vy, vt] = baseline[0]
                    indx = 0
                    for taux in range(-L_x, L_x+1):
                        indy = 0
                        for tauy in range(-L_y, L_y+1):
                            for taut in range(self.L_t):
                                if vx - taux >= 0 and  vy - tauy >= 0 and vt-taut >= 0\
                                    and vx - taux < G_S_x and vy - tauy < G_S_y\
                                        and vt - taut < G_T:
                                    intensity_values[0, vx, vy, vt] += alpha[0, 0]\
                                        * kernel_values[0, 0, indx, indy, taut]\
                                        * events_grid[0, vx - taux, vy - tauy, vt-taut]
                            indy += 1
                        indx += 1
        
        return intensity_values



def truncated_gaussian_spatial(kernel_params, space_values_x, space_values_y, delta,
                               lower, upper):
    """Truncated Gaussian spatial kernel normalized on [0, 1].

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: m and sigma.

    space_values_x : `tensor`, shape (L_X,)
        Given discretization.

    space_values_y : `tensor`, shape (L_Y,)
        Given discretization.

    delta : `array`, shape (2,)
        Step size of the discretization.

    lower : `float, default=0`
        Left bound of the support of the kernel. It should be between [0, 1].

    upper : `float, default=1`
        Right bound of the support of the kernel. It should be between [0, 1].

    Returns
    ----------
    values : `tensor`, shape (n_dim, n_dim, L_X, L_Y)
        Kernels evaluated on ``space_values``.
    """
    check_params(kernel_params, 2)
    m, sigma = kernel_params
    n_space_values_x = space_values_x.shape[0]
    n_space_values_y = space_values_y.shape[0]
    n_dim = len(sigma)
    
    values_ = torch.zeros(n_dim, n_dim, n_space_values_x, n_space_values_y)

    for i in range(n_dim):
        for j in range(n_dim):
            values_[i,j]=torch.exp((-(torch.square(space_values_x.unsqueeze(1)-m[i, j]) 
                                        + torch.square(space_values_y - m[i, j]))
                                    / (2 * torch.square(sigma[i, j]))))

    values = kernel_normalization_space_uni(values_, space_values_x, space_values_y,
                                             delta, lower, upper)
    
    return values


def grad_truncated_gaussian_spatial(kernel_params, space_values_x, space_values_y,
                                    delta, L):
    """Gradients of the Truncated Gaussian spatial kernel.

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: m and sigma.

    space_values_x : `tensor`, shape (L_X,)
        Given discretization.

    space_values_y : `tensor`, shape (L_Y,)
        Given discretization.

    delta : `array`, shape (2,)
        Step size of the discretization.

    L : `array`, shape(2,)
        Size of the kernel discretization.

    Returns
    ----------
    grad_m, grad_sigma : `tensors`
        Gradients evaluated on ``space_values``.
    """
    delta_x = delta[0]
    delta_y = delta[1]
    L_x = L[0]
    L_y = L[1]
    m, sigma = kernel_params
    n_dim = len(sigma)

    grad_m = torch.zeros(n_dim, n_dim, L_x, L_y)
    grad_sigma = torch.zeros(n_dim, n_dim, L_x, L_y)
    for i in range(n_dim):
        for j in range(n_dim):
            function = torch.exp((- (torch.square(space_values_x.unsqueeze(1) - m[i, j]) 
                                          + torch.square(space_values_y - m[i, j]))
                                       / (2 * torch.square(sigma[i, j]))))

            grad_function_mu = ((space_values_x.unsqueeze(1) + space_values_y
                                 - 2 * m[i, j]) / (torch.square(sigma[i, j])))\
                                    * function

            grad_function_s = ((torch.square(space_values_x.unsqueeze(1) - m[i, j]) 
                                          + torch.square(space_values_y - m[i, j])) /
                               (torch.pow(sigma[i, j], 3))) * function

            grad_m[i, j] = kernel_deriv_norm(function, grad_function_mu, 
                                             delta_x*delta_y)
            grad_sigma[i, j] = kernel_deriv_norm(function, grad_function_s, 
                                                 delta_x*delta_y)

    return grad_m, grad_sigma


def truncated_inv_power_law_spatial(kernel_params, space_values_x, space_values_y,
                                    delta, lower, upper):
    
    """Truncated inverse power law spatial kernel normalized on [0, 1].

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels.

    space_values_x : `tensor`, shape (L_X,)
        Given discretization.

    space_values_y : `tensor`, shape (L_Y,)
        Given discretization.

    delta : `array`, shape (2,)
        Step size of the discretization.

    lower : `float, default=0`
        Left bound of the support of the kernel. It should be between [0, 1].

    upper : `float, default=1`
        Right bound of the support of the kernel. It should be between [0, 1].

    Returns
    ----------
    values : `tensor`, shape (n_dim, n_dim, L_X, L_Y)
        Kernels evaluated on ``space_values``.
    """
    check_params(kernel_params, 2)
    m, sigma = kernel_params
    n_space_values_x = space_values_x.shape[0]
    n_space_values_y = space_values_y.shape[0]
    n_dim = len(sigma)

    q = -3/2
    
    values_ = torch.zeros(n_dim, n_dim, n_space_values_x, n_space_values_y)

    for i in range(n_dim):
        for j in range(n_dim):
            values_[i, j]=torch.pow((1+(torch.square(space_values_x.unsqueeze(1)-m[i,j]) 
                                        + torch.square(space_values_y - m[i, j]))
                                        / (sigma[i, j])), q)

    values = kernel_normalization_space_uni(values_, space_values_x, space_values_y,
                                            delta, lower, upper)
    
    return values


def grad_truncated_inv_power_law_spatial(kernel_params, space_values_x, space_values_y,
                                         delta, L):
    """Gradients of the Truncated inverse power law spatial kernel.

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: m and sigma.

    space_values_x : `tensor`, shape (L_X,)
        Given discretization.

    space_values_y : `tensor`, shape (L_Y,)
        Given discretization.

    delta : `array`, shape (2,)
        Step size of the discretization.

    L : `array`, shape(2,)
        Size of the kernel discretization.

    Returns
    ----------
    grad_m, grad_sigma : `tensors`
        Gradients evaluated on ``space_values``.
    """
    delta_x = delta[0]
    delta_y = delta[1]
    L_x = L[0]
    L_y = L[1]
    m, sigma = kernel_params
    n_dim = len(sigma)

    q = -3/2

    grad_m = torch.zeros(n_dim, n_dim, L_x, L_y)
    grad_sigma = torch.zeros(n_dim, n_dim, L_x, L_y)
    for i in range(n_dim):
        for j in range(n_dim):
            function_ = torch.pow((1+(torch.square(space_values_x.unsqueeze(1) - m[i,j]) 
                                        + torch.square(space_values_y - m[i, j]))
                                        / (sigma[i, j])), q-1)

            grad_function_mu = ((-q)*(2*space_values_x.unsqueeze(1) + 2*space_values_y
                                 - 4 * m[i, j]) / (sigma[i, j]))\
                                    * function_

            grad_function_s = ((-q)*(torch.square(space_values_x.unsqueeze(1) - m[i, j]) 
                                          + torch.square(space_values_y - m[i, j])) /
                               (torch.square(sigma[i, j]))) * function_

            function = torch.pow((1+(torch.square(space_values_x.unsqueeze(1) - m[i,j]) 
                                        + torch.square(space_values_y - m[i, j]))
                                        / (sigma[i, j])), q)
            grad_m[i, j] = kernel_deriv_norm(function, grad_function_mu,
                                              delta_x*delta_y)
            grad_sigma[i, j] = kernel_deriv_norm(function, grad_function_s,
                                                  delta_x*delta_y)

    return grad_m, grad_sigma


def truncated_exponential_temporal(kernel_params, time_values, delta, upper=1.):
    """Truncated exponential temporal kernel normalized on [0, 1].

    Parameters
    ----------
    kernel_params : `list` of size 1 of tensor of shape (n_dim, n_dim)
        Parameters of the kernel: decay.

    time_values : `tensor`, shape (L,)
        Given discretization.

    delta : `float`
        Step size of the discretization.

    upper : `float, default=1`
        Right bound of the support of the kernel. It should be between [0, 1].

    Returns
    ----------
    values : `tensor`, shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    check_params(kernel_params, 1)
    decay = kernel_params[0]

    values_ = decay.unsqueeze(2) * torch.exp(-decay.unsqueeze(2) * time_values)

    values = kernel_normalization_time(values_, time_values, delta,
                                  lower=0., upper=upper)

    return values


def grad_truncated_exponential_temporal(kernel_params, time_values, delta):
    """Gradients of the truncated exponential temporal kernel.

    Parameters
    ----------
    kernel_params : `list` of size 1 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: decay.

    time_values : `tensor`, shape (L,)
        Given discretization.

    L : `int`
        Size of the kernel discretization.

    Returns
    ----------
    grad_decay :  tensor
        Gradient evaluated on ``time_values``.
    """
    decay = kernel_params[0]
    function = decay.unsqueeze(2) * torch.exp(-decay.unsqueeze(2) * time_values)

    grad_function = (1 - decay.unsqueeze(2)
                     * time_values) * torch.exp(-decay.unsqueeze(2) * time_values)

    function[:, :, 0] = 0.
    grad_function[:, :, 0] = 0.
    function_sum = function.sum(2)[:, :, None] * delta
    grad_function_sum = grad_function.sum(2)[:, :, None] * delta
    grad_decay = (grad_function * function_sum -
                  function * grad_function_sum) / (function_sum**2)

    return grad_decay

def truncated_gaussian_temporal(kernel_params, time_values, delta, upper=1.):
    """Truncated Gaussian temporal kernel normalized on [0, 1].

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: m and sigma.

    time_values : `tensor`, shape (L,)
        Given discretization.

    delta : `float`
        Step size of the discretization.

    lower : `float, default=0`
        Left bound of the support of the kernel. It should be between [0, 1].

    upper : `float, default=1`
        Right bound of the support of the kernel. It should be between [0, 1].
    
    Returns
    ----------
    values : `tensor`, shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    check_params(kernel_params, 2)
    m, sigma = kernel_params
    n_time_values = len(time_values)
    n_dim = len(sigma)

    values_ = torch.zeros(n_dim, n_dim, n_time_values)

    for i in range(n_dim):
        for j in range(n_dim):
            values_[i, j] = torch.exp((- torch.square(time_values - m[i, j])
                                       / (2 * torch.square(sigma[i, j]))))

    values = kernel_normalization_time(values_, time_values, delta,
                                  lower=0, upper=upper)

    return values


def grad_truncated_gaussian_temporal(kernel_params, time_values, delta, L):
    """Gradients of the truncated Gaussian temporal kernel.

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: m and sigma.

    time_values : `tensor`, shape (L,)
        Given discretization.

    L : `int`
        Size of the kernel discretization.

    Returns
    ----------
    grad_m, grad_sigma: `tensors`
        Gradients evaluated on ``time_values``.
    """
    m, sigma = kernel_params
    n_dim = len(sigma)

    grad_m = torch.zeros(n_dim, n_dim, L)
    grad_sigma = torch.zeros(n_dim, n_dim, L)
    for i in range(n_dim):
        for j in range(n_dim):
            function = torch.exp((- torch.square(time_values - m[i, j]) /
                                 (2 * torch.square(sigma[i, j]))))

            grad_function_mu = ((time_values - m[i, j]) / (torch.square(sigma[i, j]))) \
                * function

            grad_function_s = (torch.square(time_values - m[i, j]) /
                               (torch.pow(sigma[i, j], 3))) * function

            grad_m[i, j] = kernel_deriv_norm(function, grad_function_mu, delta)
            grad_sigma[i, j] = kernel_deriv_norm(function, grad_function_s, delta)

    return grad_m, grad_sigma


def kumaraswamy_temporal(kernel_params, time_values):
    """Kumaraswamy temporal kernel.

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: u and sigma.

    time_values : `tensor`, shape (L,)
        Given discretization.

    Returns
    ----------
    values : `tensor`, shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    check_params(kernel_params, 2)
    a, b = kernel_params
    n_dim, _ = a.shape

    values = torch.zeros(n_dim, n_dim, len(time_values))
    for i in range(n_dim):
        for j in range(n_dim):
            pa = a[i, j] - 1
            pb = b[i, j] - 1
            values[i, j] = (a[i, j] * b[i, j] * (time_values**pa)
                            * ((1 - time_values**a[i, j]) ** pb))
            mask_kernel = (time_values <= 0.) | (time_values >= 1.)
            values[i, j, mask_kernel] = 0.

    return values


def grad_kumaraswamy_temporal(kernel_params, time_values, L):
    """Gradients of the Kumaraswamy temporal kernel.

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: u and sigma.

    time_values : `tensor`, shape (L,)
        Given discretization.

    L : `int`
        Size of the kernel discretization.

    Returns
    ----------
    grad_a, grad_b : `tensors`
        Gradients evaluated on ``time_values``.
    """
    a, b = kernel_params
    n_dim, _ = a.shape
    kernel_values = kumaraswamy_temporal(kernel_params, time_values)
    b_minusone = b - 1
    kernel_params_ = [a, b_minusone]
    kernel_values_ = kumaraswamy_temporal(kernel_params_, time_values)

    grad_a = torch.zeros(n_dim, n_dim, L)
    grad_b = torch.zeros(n_dim, n_dim, L)
    for i in range(n_dim):
        for j in range(n_dim):
            grad_a[i, j] = kernel_values[i, j] * (1 / a[i, j] + torch.log(time_values))\
                - kernel_values_[i, j] * torch.log(time_values) * time_values**a[i, j]
            grad_b[i, j] = kernel_values[i, j] * (1 / b[i, j]
                                                  + torch.log(1 - time_values**a[i, j]))
            mask_kernel = (time_values <= 0.) | (time_values >= 1.)
            grad_a[i, j, mask_kernel] = 0.
            grad_b[i, j, mask_kernel] = 0.

    return grad_a, grad_b

def raised_cosine_temporal(kernel_params, time_values):
    """Raised Cosine temporal kernel.

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: u and sigma.

    time_values : `tensor`, shape (L,)
        Given discretization.

    Returns
    ----------
    values : `tensor`, shape (n_dim, n_dim, L)
        Kernels evaluated on ``time_values``.
    """
    # reparam: alpha= alpha' / (2*sigma)
    check_params(kernel_params, 2)
    u, sigma = kernel_params
    n_dim, _ = u.shape

    values = torch.zeros(n_dim, n_dim, len(time_values))
    for i in range(n_dim):
        for j in range(n_dim):
            values[i, j] = (1 + torch.cos(((time_values - u[i, j]) / sigma[i, j]
                                          * np.pi) - np.pi))

            mask_kernel = (time_values < u[i, j]) | (
                time_values > (u[i, j] + 2 * sigma[i, j]))
            values[i, j, mask_kernel] = 0.

    return values


def grad_raised_cosine_temporal(kernel_params, time_values, L):
    """Gradients of the Raised Cosine temporal kernel.

    Parameters
    ----------
    kernel_params : `list` of size 2 of tensor of shape (n_dim, n_dim)
        Parameters of the kernels: u and sigma.

    time_values : `tensor`, shape (L,)
        Given discretization.

    L : `int`
        Size of the kernel discretization.

    Returns
    ----------
    grad_u, grad_sigma : tensors
        Gradients evaluated on ``time_values``.
    """
    u, sigma = kernel_params
    n_dim, _ = u.shape

    grad_u = torch.zeros(n_dim, n_dim, L)
    grad_sigma = torch.zeros(n_dim, n_dim, L)
    for i in range(n_dim):
        for j in range(n_dim):
            temp_1 = ((time_values - u[i, j]) / sigma[i, j])
            temp_2 = temp_1 * np.pi - np.pi
            grad_u[i, j] = np.pi * torch.sin(temp_2) / sigma[i, j]
            grad_sigma[i, j] = (np.pi * temp_1 / sigma[i, j]**2) * torch.sin(temp_2)
            mask_grad = (time_values < u[i, j]) | (
                time_values > (u[i, j] + 2 * sigma[i, j]))
            grad_u[i, j, mask_grad] = 0.
            grad_sigma[i, j, mask_grad] = 0.

    return grad_u, grad_sigma