import numpy as np
import torch
from scipy.optimize import minimize


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def find_max_spatial(intensity_function, support):
    """Find the maximum intensity of a function with two argues."""
    s_init = [(support[0][0] + support[0][1])/2, (support[1][0] + support[1][1])/2]
    res = minimize(lambda s: -intensity_function(s), s_init, bounds=support)
    return -res.fun


def scaled_gaussian(s, scale=100, sigma=0.5):
    x, y = s
    return scale * (
        1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)

def kernel_normalization_space_uni(kernel_values, 
                               space_values_x, space_values_y, delta, lower, upper):
    """Normalize the given kernel on the given discrete grid.
    """
    space_values = np.stack((space_values_x, space_values_y), axis = -1)
    kernel_norm = kernel_values.clone()
    mask_kernel_x = (space_values[:, 0] < lower[0]) | (space_values[:, 0] > upper[0])
    mask_kernel_y = (space_values[:, 1] < lower[1]) | (space_values[:, 1] > upper[1])
    kernel_norm[:, :, mask_kernel_x, mask_kernel_y] = 0.

    sum = kernel_norm.sum((2, 3))[:, :, None, None]

    kernel_norm /= (sum * delta[0]*delta[1])

    return kernel_norm


def kernel_normalization_time(kernel_values, time_values, delta, lower=0, upper=1):
    """Normalize the given kernel on the given discrete grid.
    """
    kernel_norm = kernel_values.clone()
    mask_kernel = (time_values <= lower) | (time_values > upper)
    kernel_norm[:, :, mask_kernel] = 0.

    kernel_norm /= (kernel_norm.sum(2)[:, :, None] * delta)

    return kernel_norm


def kernel_deriv_norm(function, grad_function_param, delta):
    """Normalize the given gradient kernels on the given discrete grid.
    """
    function[0] = 0.
    grad_function_param[0] = 0.

    function_sum = function.sum() * delta
    grad_function_param_sum = grad_function_param.sum() * delta

    return (function_sum * grad_function_param -
            function * grad_function_param_sum) / (function_sum**2)


def kernel_deriv_norm_mean(function, grad_function_param, delta):
    """Normalize the given gradient kernels on the given discrete grid.
    """
    function[0] = 0.
    grad_function_param[0] = 0.

    function_sum = function.sum() * delta
    grad_function_param_sum = grad_function_param.sum() * delta
    
    return (function_sum * grad_function_param -
            function[:, :, np.newaxis] * grad_function_param_sum) / (function_sum**2)


def kernel_deriv_norm_sigma(function, grad_function_param, delta):
    """Normalize the given gradient kernels on the given discrete grid.
    """
    function[0] = 0.
    grad_function_param[0] = 0.

    function_sum = function.sum() * delta
    grad_function_param_sum = grad_function_param.sum() * delta
    
    return (function_sum * grad_function_param -
            function[:, :, np.newaxis, np.newaxis] * grad_function_param_sum) / \
                (function_sum**2)


def check_params(list_params, number_params):
    """Check if the list of parameters is equal to the number of parameters.
    """
    if len(list_params) != number_params:
        raise Exception("The number of parameters for this kernel\
                         should be equal to {}".format(number_params))
    return 0


def multidim_cumsum(a):
    out = a.cumsum(-1)
    for i in range(2, a.ndim+1):
        np.cumsum(out, axis=-i, out=out)
    return out


def optimizer(param, params_optim, solver='RMSprop'):
    """Set the Pytorch optimizer.

    Parameters
    ----------
    param : XXX
    lr : float
        learning rate
    solver : str
        solver name, possible values are 'GD', 'RMSProp', 'Adam'
        or 'CG'
    Returns
    -------
    XXX
    """
    if solver == 'GD':
        return torch.optim.SGD(param, **params_optim)
    elif solver == 'RMSprop':
        return torch.optim.RMSprop(param, **params_optim)
    elif solver == 'Adam':
        return torch.optim.Adam(param, **params_optim)
    else:
        raise NotImplementedError(
            "solver must be 'GD', 'RMSProp', 'Adam'," f"got '{solver}'")