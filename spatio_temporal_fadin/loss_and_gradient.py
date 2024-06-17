import torch
import numpy as np
import time

def negative_log_likelihood(intensity, events_grid, delta):
    """Compute the negative log likelihood.

    Parameters
    ----------
    intensity : tensor, shape (n_dim, n_grid_spatio_x, n_grid_spatio_y, n_grid_time)
        Values of the intensity function evaluated  on the grid.

    events_grid : tensor, shape (n_dim, n_grid_spatio_x, n_grid_spatio_y, n_grid_time)
        Events projected on the pre-defined grid.

    delta : list of float
        Step size of the discretization grids.
    """
    delta_spatio = delta[0]
    delta_time = delta[1]

    temp = intensity * events_grid

    temp2 = temp[temp>0]

    return 2*(((intensity).sum((1, 2, 3)) * delta_spatio[0]*delta_spatio[1]*delta_time -
                 np.log(temp2).sum()).sum()) / events_grid.sum()

def discrete_l2_loss_conv(intensity, events_grid, grid_step):
    """Compute the l2 discrete loss using convolutions.

    Parameters
    ----------
    intensity : tensor, shape (n_dim, n_grid_spatio_x, n_grid_spatio_y, n_grid_time)
        Values of the intensity function evaluated  on the grid.

    events_grid : tensor, shape (n_dim, n_grid_spatio_x, n_grid_spatio_y, n_grid_time)
        Events projected on the pre-defined grid.

    delta : list of float
        Step size of the discretization grids.
    """
    grid_step_spatio = grid_step[0]
    grid_step_time = grid_step[1]
    
    return 2 * (((intensity**2).sum((1, 2, 3))*0.5*grid_step_spatio**2*grid_step_time -
                 (intensity * events_grid).sum((1, 2, 3))).sum()) / events_grid.sum()

def squared_compensator_1(baseline):
    """Compute the value of the first term of the
    discrete l2 loss using precomputations

    .. math::
        ||\\mu||_2^2

    Parameters
    ----------
    baseline : tensor, shape (n_dim,)
    """
    return torch.linalg.norm(baseline, ord=2) ** 2


def squared_compensator_2(zG, baseline, alpha, kernel):
    """Compute the value of the second term of the
    discrete l2 loss using precomputations

    Parameters
    ----------
    zG : tensor, shape (n_dim, L_X, L_Y, L_T)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)
        Kernel values on the discretization.
    """
    n_dim = zG.shape[0]

    if n_dim > 1:
        prod_zG_ker = torch.einsum('juvt,ijuvt->ij', zG, kernel)
        alpha_prod = alpha * prod_zG_ker
        res = torch.dot(baseline, alpha_prod.sum(1))
    else:
        res_pre = ((zG[0] * kernel[0, 0]).sum())
        res = baseline[0] * (alpha[0, 0] * res_pre.cpu())

    return res


def squared_compensator_3(ztzG, alpha, kernel):
    """Compute the value of the third term of the
    discrete l2 loss using precomputations

    Parameters
    ----------
    ztzG : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)
        Kernel values on the discretization.
    """
    n_dim, _, L_S, _, L_T = kernel.shape

    if n_dim == 1:

        kernel1 = kernel[0][0]
        kernel2 = kernel[0][0]
        
        res_pre = (torch.einsum('uvw,xyz->uvwxyz', kernel1, kernel2) * ztzG).sum()

        res = alpha[0, 0] * alpha[0, 0] * res_pre.cpu()
        
    else:
    
        temp_res = torch.zeros(n_dim, n_dim, n_dim)

        for i in range(n_dim):
            for j in range(n_dim):
                for k in range(n_dim):

                    kernel1 = kernel[i][j]
                    kernel2 = kernel[i][k]
                    
                    B2 = ztzG[j][k]
                    
                    temp1 = torch.einsum('uvw,xyz->uvwxyz', kernel1, kernel2)
                    temp2 = torch.einsum('uvwxyz,uvwxyz->uvwxyz', temp1, B2)

                    temp_res[i, j, k] = alpha[i, j] * alpha[i, k] * temp2.sum()

        res = temp_res.sum()

    return res

def intens_events(zN, baseline, alpha, kernel, n_events):
    """Compute the value of the 4th term of the
    discrete l2 loss using precomputations, i.e.
    the intensity function values evaluated in the events.

    Parameters
    ----------
    zN : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)
        Kernel values on the discretization.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.
    """
    n_dim = kernel.shape[0]

    if n_dim > 1:
        prod_zN_ker = torch.einsum('ijuvt,ijuvt->ij', zN, kernel)
        alpha_prod_dot = alpha * prod_zN_ker
        base_ev = torch.dot(baseline, n_events)
        res = base_ev + alpha_prod_dot.sum()
    else:
        res_pre = ((zN[0, 0] * kernel[0, 0]).sum())
        res = baseline[0] * n_events[0] + res_pre.cpu() * alpha[0, 0]
        
    return res

def discrete_l2_loss_precomputation_RD(zG, zN, ztzG, baseline, alpha, kernel, n_events,
                                       n_events_tensor, delta, end_time, spatio_bound):
    """Compute the l2 discrete loss using precomputation terms.

    Parameters
    ----------
    zG : tensor, shape (n_dim, L_X, L_Y, L_T)

    zN : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)

    ztzG : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T, L_X, L_Y, L_T)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L_S, L_S, L_T)
        Kernel values on the discretization.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    delta : float
        Step size of the discretization grid.

    end_time : float
        The end time of the Hawkes process.
    """
    
    delta_S = delta[0]
    delta_T = delta[1]

    comp_1 = (end_time + delta_T) * (2 * spatio_bound[0] + delta_S[0])\
          * (2 * spatio_bound[1] + delta_S[1]) * squared_compensator_1(baseline)
    comp_2 = 2 * delta_S[0] * delta_S[1] * delta_T\
          * squared_compensator_2(zG, baseline, alpha, kernel)
    comp_3 = delta_S[0] * delta_S[1] * delta_T\
          * squared_compensator_3(ztzG, alpha, kernel)
    intens_ev = 2 * intens_events(zN, baseline, alpha, kernel, n_events_tensor)

    loss_precomp = comp_1 + comp_2 + comp_3 - intens_ev

    return loss_precomp / n_events.sum()











def get_grad_baseline(zG, baseline, alpha, kernel,
                      delta, n_events, end_time, spatio_bound):
    """Return the gradient of the discrete l2 loss w.r.t. the baseline.

    Parameters
    ----------
    zG : tensor, shape (n_dim, L_X, L_Y, L_T)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)
        Kernel values on the discretization.

    delta : 'array', shape (2,)
        Spatial (shape (2,)) and temporal step sizes of the discretization.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.
        
    end_time : `int`
        T the stopping time of the MSTH process

    Returns
    ----------
    grad_baseline: tensor, shape (dim,)
    """

    delta_s = delta[0]
    delta_t = delta[1]

    term1 = 2 * (end_time + delta_t) * (2 * spatio_bound[0] + delta_s[0]) * (2 * spatio_bound[1] + delta_s[1]) * baseline # 2 * (end_time) * (2 * spatio_bound)**2 * baseline
    term2_pre = torch.einsum('kuvt,mkuvt->mk', zG, kernel)
    term2 = 2 * delta_s[0]*delta_s[1] * delta_t * alpha * term2_pre.cpu()
    term3 = 2 * n_events

    grad_baseline = (term1 + term2 - term3) / n_events

    return grad_baseline




def get_grad_eta_gaussian(zG, ztzG, zN, baseline, alpha, kernel,
                 grad_kernel_mean, grad_kernel_sigma, delta, n_events):
    """Return the gradient of the discrete l2 loss w.r.t. one kernel parameters.

    Parameters
    ----------
    zG : tensor, shape (n_dim, L_X, L_Y, L_T)

    zN : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)

    ztzG : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T, L_X, L_Y, L_T)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)
        Kernel values on the discretization.

    grad_kernel : list of tensor of shape (n_dim, n_dim, L_X, L_Y, L_T)
        Gradient values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_theta : tensor, shape (n_dim, n_dim)
    """

    delta_s = delta[0]
    delta_t = delta[1]

    term1_mean_pre = torch.einsum('luvt,mluvt->ml', zG, grad_kernel_mean)
    term1_mean = 2 * delta_s[0]*delta_s[1] * delta_t * baseline * alpha * term1_mean_pre.cpu()
    term1_sigma_pre = torch.einsum('luvt,mluvt->ml', zG, grad_kernel_sigma)
    term1_sigma = 2 * delta_s[0]*delta_s[1] * delta_t * baseline * alpha * term1_sigma_pre.cpu()

    grad_kernel1_mean = grad_kernel_mean[0][0]
    grad_kernel1_sigma = grad_kernel_sigma[0][0]
    kernel2 = kernel[0][0]
    
    term2_mean_pre = (torch.einsum('uvw,xyz->uvwxyz', grad_kernel1_mean, kernel2) * ztzG).sum()
    term2_mean = delta_s[0]*delta_s[1] * delta_t * alpha * alpha * term2_mean_pre.cpu()
    term2_sigma_pre = (torch.einsum('uvw,xyz->uvwxyz', grad_kernel1_sigma, kernel2) * ztzG).sum()
    term2_sigma = delta_s[0]*delta_s[1] * delta_t * alpha * alpha * term2_sigma_pre.cpu()
    
    term3_mean_pre = torch.einsum('mluvt,mluvt->ml', zN, grad_kernel_mean)
    term3_mean = 2 * alpha * term3_mean_pre.cpu()
    term3_sigma_pre = torch.einsum('mluvt,mluvt->ml', zN, grad_kernel_sigma)
    term3_sigma = 2 * alpha * term3_sigma_pre.cpu()
    
    grad_eta_mean = (term1_mean + 2*term2_mean - term3_mean) / n_events.sum()
    grad_eta_sigma = (term1_sigma + 2*term2_sigma - term3_sigma) / n_events.sum()

    return grad_eta_mean, grad_eta_sigma

def get_grad_eta(zG, ztzG, zN, baseline, alpha, kernel,
                 grad_kernel, delta, n_events):
    """Return the gradient of the discrete l2 loss w.r.t. one kernel parameters.

    Parameters
    ----------
    zG : tensor, shape (n_dim, L_X, L_Y, L_T)

    zN : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)

    ztzG : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T, L_X, L_Y, L_T)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)
        Kernel values on the discretization.

    grad_kernel : list of tensor of shape (n_dim, n_dim, L_X, L_Y, L_T)
        Gradient values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_theta : tensor, shape (n_dim, n_dim)
    """

    delta_s = delta[0]
    delta_t = delta[1]
    
    term1_pre = torch.einsum('luvt,mluvt->ml', zG, grad_kernel)
    term1 = 2 * delta_s[0]*delta_s[1] * delta_t * baseline * alpha * term1_pre.cpu()

    grad_kernel1 = grad_kernel[0][0]
    kernel2 = kernel[0][0]
    
    term2_pre = (torch.einsum('uvw,xyz->uvwxyz', grad_kernel1, kernel2) * ztzG).sum()
    term2 = delta_s[0]*delta_s[1] * delta_t * alpha * alpha * term2_pre.cpu()

    term3_pre = torch.einsum('mluvt,mluvt->ml', zN, grad_kernel)
    term3 = 2 * alpha * term3_pre.cpu()

    grad_eta = (term1 + 2*term2 - term3) / n_events.sum()

    return grad_eta





def get_grad_alpha(zG, ztzG, zN, baseline, alpha, kernel, delta, n_events):
    """Return the gradient of the discrete l2 loss w.r.t. one kernel parameters.

    Parameters
    ----------
    zG : tensor, shape (n_dim, L_X, L_Y, L_T)

    zN : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)

    ztzG : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T, L_X, L_Y, L_T)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L_X, L_Y, L_T)
        Kernel values on the discretization.

    grad_kernel : list of tensor of shape (n_dim, n_dim, L_X, L_Y, L_T)
        Gradient values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_theta : tensor, shape (n_dim, n_dim)
    """

    delta_s = delta[0]
    delta_t = delta[1]

    term1_pre = torch.einsum('luvt,mluvt->ml', zG, kernel)
    term1 = 2 * delta_s[0]*delta_s[1] * delta_t * baseline * term1_pre.cpu()

    kernel1 = kernel[0][0]
    kernel2 = kernel[0][0]

    term2_pre = (torch.einsum('uvw,xyz->uvwxyz', kernel1, kernel2) * ztzG).sum()
    term2 = 2 * alpha * delta_s[0]*delta_s[1] * delta_t * term2_pre.cpu()
    
    term3_pre = 2*(zN*kernel).sum()
    term3 = term3_pre.cpu()

    grad_alpha = (term1 + term2 - term3) / n_events
    
    return grad_alpha