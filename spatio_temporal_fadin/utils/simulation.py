import numpy as np
import torch
import scipy.stats
from scipy.interpolate import interp1d
from scipy.integrate import simps

from spatio_temporal_fadin.utils.functions import *


def simu_multi_spatial_poison(intensity, spatial_support, end_time,
                              upper_bound=None, random_state=None):
    """Simulate multivariate Spatial Poisson processes on support with
    the Ogata's modified thinning algorithm by superposition of univariate processes.
    If the intensity is a numerical value, simulate a Homegenous Spatial Poiss Process,
    If the intensity is a function, simulate an Inhomogenous Spatial Poisson Process.

    Parameters
    ----------
    intensity: list of callable, list of int or float
        the intensity function of the underlying Spatial Poisson process.
        If callable, a inhomogenous Poisson process is simulated.
        If int or float, an homogenous Poisson process is simulated.

    spatial_support : list of dim 2 of list of dim 2 | dtype: float
        Support of the Spatial Poisson process.
    
    end_time : int | float
        Duration of the Poisson process.

    upper_bound : int, float or None, default=None
        Upper bound of the intensity functions. If None,
        the maximum of the function is taken onto a finite discrete grid.

    random_state : int, RandomState instance or None, default=None
        Set the numpy seed to 'random_state'.

    Returns
    -------
    events : list of array
        The timestamps of the spatial point process' events.
    """

    rng = check_random_state(random_state)

    events = []
    n_dim = len(intensity)

    xmin, xmax = spatial_support[0]
    ymin, ymax = spatial_support[1]
    x_delta = xmax - xmin
    y_delta = ymax - ymin
    spatial_area = x_delta * y_delta

    if not callable(intensity[0]):
        assert isinstance(intensity[0], (int, float))
        # check if the number of event has to depend on time or not
        n_events = rng.poisson(end_time*spatial_area*np.array(intensity), size=n_dim)
        for i in range(n_dim):
            xx = rng.uniform(0, x_delta, ((n_events[i], 1))) + xmin
            yy = rng.uniform(0, y_delta, ((n_events[i], 1))) + ymin
            spatial_evi = np.array([xx, yy]).squeeze().T
            events_time = rng.uniform(0, end_time, spatial_evi.shape[0])
            sorted_events_time = np.sort(events_time)
            ev = np.concatenate((spatial_evi, sorted_events_time[:, None]), axis=1)
            events.append(ev)
        return events

    if upper_bound is None:
        upper_bound = np.zeros(n_dim)
        for i in range(n_dim):
            upper_bound[i] = find_max_spatial(intensity[i], spatial_support)

    # Simulate a Poisson point process
    n_events = rng.poisson(lam=end_time*upper_bound*spatial_area, size=n_dim)
    for i in range(n_dim):
        xx = rng.uniform(0, x_delta, ((n_events[i], 1))) + xmin
        yy = rng.uniform(0, y_delta, ((n_events[i], 1))) + ymin
        candidate_evi = np.array([xx, yy]).squeeze()

        accepted_evi = rng.uniform(
            0, upper_bound[i], n_events[i]) < intensity[i](candidate_evi)
        spatial_evi = candidate_evi[:, accepted_evi].T
        events_time = rng.uniform(0, end_time, spatial_evi.shape[0])
        sorted_events_time = np.sort(events_time)
        ev = np.concatenate((spatial_evi, sorted_events_time[:, None]), axis=1)
        events.append(ev)

    return events


def simu_spatial_hawkes_cluster(end_time, support, baseline, alpha, time_kernel,
                                space_x_kernel, space_y_kernel,
                                params_time_kernel=dict(), params_space_x_kernel=dict(),
                                params_space_y_kernel=dict(), random_state=None):
    """ Simulate a multivariate spatio-temporal Hawkes process following an
       immigration-birth procedure.

    Parameters
    ----------
    end_time : int | float
        Duration of the Poisson process.

    support : list of list of size 2x2
        Spatial support

    baseline : callable or array of float of size (n_dim,)
        Baseline parameter of the Hawkes process.

    alpha : array of float of size (n_dim, n_dim)
        Weight parameter associated to the kernel function.

    time_kernel: str
        The choice of the time kernel for the simulation.
        Kernel available are probability distribution from scipy.stats module.

    space_x_kernel: str
        The choice of the first coordinate space kernel for the simulation.
        Kernel available are probability distribution from scipy.stats module.

    space_y_kernel: str
        The choice of the second coordinate space kernel for the simulation.
        Kernel available are probability distribution from scipy.stats module.

    params_time_kernel: dict
        Parameters of the time kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions.

    params_space_x_kernel: dict
        Parameters of the first coordinate space kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions.

    params_space_y_kernel: dict
        Parameters of the second coordinate space kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions.

    random_state : int, RandomState instance or None, default=None
        Set the numpy seed to 'random_state'.

    Returns
    -------
    events : list of array-like with shape (n_events, 3)
        The timestamps of the point process' events.
    """
    rng = check_random_state(random_state)

    n_dim = alpha.shape[0]
    immigrants = simu_multi_spatial_poison(baseline, support, end_time, random_state=1)

    immigrants_x = [immigrants[i][:, 0] for i in range(n_dim)]
    immigrants_y = [immigrants[i][:, 1] for i in range(n_dim)]
    immigrants_time = [immigrants[i][:, 2] for i in range(n_dim)]

    gen_x = dict(gen0=immigrants_x)
    gen_y = dict(gen0=immigrants_y)
    gen_time = dict(gen0=immigrants_time)

    events = immigrants.copy()

    sample_from_time_kernel = getattr(scipy.stats, time_kernel)
    sample_from_space_x_kernel = getattr(scipy.stats, space_x_kernel)
    sample_from_space_y_kernel = getattr(scipy.stats, space_y_kernel)

    it = 0
    while len(gen_time[f'gen{it}']):
        # print(f"Simulate generation {it}\r")
        Ck_time = gen_time[f'gen{it}']
        Ck_x = gen_x[f'gen{it}']
        Ck_y = gen_y[f'gen{it}']

        Dk = [[0] * n_dim for _ in range(n_dim)]
        C_time = [[0] * n_dim for _ in range(n_dim)]
        C_x = [[0] * n_dim for _ in range(n_dim)]
        C_y = [[0] * n_dim for _ in range(n_dim)]
        F_time = []
        F_x = []
        F_y = []
        s = 0
        for i in range(n_dim):
            Fi_time = []
            Fi_x = []
            Fi_y = []
            for j in range(n_dim):
                Dk[i][j] = rng.poisson(lam=alpha[i, j], size=len(Ck_time[j]))
                nij = Dk[i][j].sum()

                C_time[i][j] = np.repeat(Ck_time[j], repeats=Dk[i][j])
                C_x[i][j] = np.repeat(Ck_x[j], repeats=Dk[i][j])
                C_y[i][j] = np.repeat(Ck_y[j], repeats=Dk[i][j])

                Eij_time = sample_from_time_kernel.rvs(**params_time_kernel, size=nij,
                                                       random_state=random_state)
                Eij_x = sample_from_space_x_kernel.rvs(
                    **params_space_x_kernel, size=nij, random_state=random_state)
                # random state is set to 10000 just to avoid redondancy if
                # distribution of x and y are the same (sampled with the same seed)
                Eij_y = sample_from_space_y_kernel.rvs(
                    **params_space_y_kernel, size=nij, random_state=10000)

                Fij_time = C_time[i][j] + Eij_time
                Fij_x = C_x[i][j] + Eij_x
                Fij_y = C_y[i][j] + Eij_y

                Fi_time.append(Fij_time)
                Fi_x.append(Fij_x)
                Fi_y.append(Fij_y)

                s += Fij_time.shape[0]
            F_time.append(np.hstack(Fi_time))
            F_x.append(np.hstack(Fi_x))
            F_y.append(np.hstack(Fi_y))

        if s > 0:
            gen_time[f'gen{it+1}'] = F_time
            gen_x[f'gen{it+1}'] = F_x
            gen_y[f'gen{it+1}'] = F_y

            for i in range(n_dim):
                F = np.vstack([F_x[i], F_y[i], F_time[i]]).T
                events[i] = np.concatenate((events[i], F))
        else:
            for i in range(n_dim):
                valid_events_x = (
                    support[0][0] < events[i][:, 0]) * (events[i][:, 0] < support[0][1])
                valid_events_y = (
                    support[1][0] < events[i][:, 1]) * (events[i][:, 1] < support[1][1])

                valid_events_time = events[i][:, 2] < end_time
                valid_events = valid_events_x * valid_events_y * valid_events_time
                valid = events[i][valid_events]
                sorting = np.argsort(valid[:, 2])
                events[i] = valid[sorting]
            break

        it += 1

    return events, immigrants


class custom_distribution(scipy.stats.rv_continuous):
    """Construct finite support density and allows efficient scipy sampling"""
    def __init__(self, custom_density, space=False, params=dict(), kernel_length=1.):
        super().__init__()
        # init our variance divergence
        self.density_name = custom_density
        self.params = params
        self.kernel_length = kernel_length
        self.is_space = space
        # init our cdf and ppf functions
        self.cdf_func, self.ppf_func = self.create_cdf_ppf()
    
    # function to normalise the pdf over chosen domain
    def normalisation(self, x):
        return simps(self._pdf(x), x)

    def create_cdf_ppf(self):
        # define normalization support with the given kernel length
        if self.is_space:
            discrete = torch.linspace(-self.kernel_length[0], self.kernel_length[1], 1001)
        else:
            discrete = torch.linspace(0, self.kernel_length, 1001)
        
        # normalise our pdf to sum to 1 so it satisfies a distribution
        norm_constant = self.normalisation(discrete)
        # compute pdfs to be summed to form cdf
        my_pdfs = self._pdf(discrete) / norm_constant
        # cumsum to form cdf
        my_cdf = np.cumsum(my_pdfs)
        # make sure cdf bounded on [0,1]
        my_cdf = my_cdf / my_cdf[-1]
        # create cdf and ppf
        func_cdf = interp1d(discrete, my_cdf)
        func_ppf = interp1d(my_cdf, discrete, fill_value='extrapolate')
        return func_cdf, func_ppf

    # pdf function for averaged normals
    def _pdf(self, x):
        return self.density_name(x, self.params)

    # cdf function
    def _cdf(self, x):
        return self.cdf_func(x)

    # inverse cdf function
    def _ppf(self, x):
        return self.ppf_func(x)


def custom_density(density, params=dict(), size=1, kernel_length=None, space=False):
    """Sample elements from custom or scipy-defined distributions"""
    distrib = custom_distribution(custom_density=density, space=space, params=params,
                                  kernel_length=kernel_length)

    return distrib.rvs(size=size)


def simu_spatial_hawkes_cluster_custom(end_time, support, baseline, alpha, time_kernel,
                                space_x_kernel, space_y_kernel,
                                params_time_kernel, params_space_x_kernel,
                                params_space_y_kernel, time_kernel_length=None,
                                spatial_kernel_length=None, delta=None, random_state=None):
    """ Simulate a multivariate spatio-temporal Hawkes process following an
       immigration-birth procedure.

    Parameters
    ----------
    end_time : int | float
        Duration of the Poisson process.

    support : list of list of size 2x2
        Spatial support

    baseline : callable or array of float of size (n_dim,)
        Baseline parameter of the Hawkes process.

    alpha : array of float of size (n_dim, n_dim)
        Weight parameter associated to the kernel function.

    time_kernel: str
        The choice of the time kernel for the simulation.
        Kernel available are probability distribution from scipy.stats module.

    space_x_kernel: str
        The choice of the first coordinate space kernel for the simulation.
        Kernel available are probability distribution from scipy.stats module.

    space_y_kernel: str
        The choice of the second coordinate space kernel for the simulation.
        Kernel available are probability distribution from scipy.stats module.

    params_time_kernel: dict
        Parameters of the time kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions.

    params_space_x_kernel: dict
        Parameters of the first coordinate space kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions.

    params_space_y_kernel: dict
        Parameters of the second coordinate space kernel used to simulate the process.
        It must follow parameters associated to scipy.stats distributions.

    time_kernel_length: int
        Temporal kernel length.

    spatial_kernel_length: array of size (2,)
        Spatial kernel length.

    random_state : int, RandomState instance or None, default=None
        Set the numpy seed to 'random_state'.

    Returns
    -------
    events : list of array-like with shape (n_events, 3)
        The timestamps of the point process' events.
    """
    rng = check_random_state(random_state)

    n_dim = alpha.shape[0]
    immigrants = simu_multi_spatial_poison(baseline, support, end_time, random_state=1)

    immigrants_x = [immigrants[i][:, 0] for i in range(n_dim)]
    immigrants_y = [immigrants[i][:, 1] for i in range(n_dim)]
    immigrants_time = [immigrants[i][:, 2] for i in range(n_dim)]

    gen_x = dict(gen0=immigrants_x)
    gen_y = dict(gen0=immigrants_y)
    gen_time = dict(gen0=immigrants_time)

    events = immigrants.copy()

    it = 0
    while len(gen_time[f'gen{it}']):
        # print(f"Simulate generation {it}\r")
        Ck_time = gen_time[f'gen{it}']
        Ck_x = gen_x[f'gen{it}']
        Ck_y = gen_y[f'gen{it}']

        Dk = [[0] * n_dim for _ in range(n_dim)]
        C_time = [[0] * n_dim for _ in range(n_dim)]
        C_x = [[0] * n_dim for _ in range(n_dim)]
        C_y = [[0] * n_dim for _ in range(n_dim)]
        F_time = []
        F_x = []
        F_y = []
        s = 0
        for i in range(n_dim):
            Fi_time = []
            Fi_x = []
            Fi_y = []
            for j in range(n_dim):
                Dk[i][j] = rng.poisson(lam=alpha[i, j], size=len(Ck_time[j]))
                nij = Dk[i][j].sum()

                C_time[i][j] = np.repeat(Ck_time[j], repeats=Dk[i][j])
                C_x[i][j] = np.repeat(Ck_x[j], repeats=Dk[i][j])
                C_y[i][j] = np.repeat(Ck_y[j], repeats=Dk[i][j])

                if time_kernel == 'norm':
                    Eij_time = custom_density(truncated_gaussian_custom_time, params_time_kernel,
                                              size=nij, kernel_length=time_kernel_length)
                elif time_kernel == 'expon':
                    Eij_time = custom_density(truncated_exponential_custom, params_time_kernel,
                                              size=nij, kernel_length=time_kernel_length)
                elif time_kernel == 'kur':
                    Eij_time = custom_density(kumaraswamy_custom, params_time_kernel,
                                              size=nij, kernel_length=time_kernel_length)
                elif time_kernel == 'cos':
                    Eij_time = custom_density(raised_cosine_custom, params_time_kernel,
                                              size=nij, kernel_length=time_kernel_length)
                
                if space_x_kernel == 'norm':
                    Eij_x = custom_density(truncated_gaussian_custom, params_space_x_kernel,
                                        size=nij, kernel_length=spatial_kernel_length, space=True)
                    
                    Eij_y = custom_density(truncated_gaussian_custom, params_space_y_kernel,
                                        size=nij, kernel_length=spatial_kernel_length, space=True)
                elif space_x_kernel == "pow":
                    Eij_x = custom_density(truncated_inv_power_law_custom, params_space_x_kernel,
                                        size=nij, kernel_length=spatial_kernel_length, space=True)
                    
                    Eij_y = custom_density(truncated_inv_power_law_custom, params_space_y_kernel,
                                        size=nij, kernel_length=spatial_kernel_length, space=True)
                
                Fij_time = C_time[i][j] + Eij_time
                Fij_x = C_x[i][j] + Eij_x
                Fij_y = C_y[i][j] + Eij_y

                Fi_time.append(Fij_time)
                Fi_x.append(Fij_x)
                Fi_y.append(Fij_y)

                s += Fij_time.shape[0]
            F_time.append(np.hstack(Fi_time))
            F_x.append(np.hstack(Fi_x))
            F_y.append(np.hstack(Fi_y))

        if s > 0:
            gen_time[f'gen{it+1}'] = F_time
            gen_x[f'gen{it+1}'] = F_x
            gen_y[f'gen{it+1}'] = F_y

            for i in range(n_dim):
                F = np.vstack([F_x[i], F_y[i], F_time[i]]).T
                events[i] = np.concatenate((events[i], F))
        else:
            for i in range(n_dim):
                valid_events_x = (
                    support[0][0] < events[i][:, 0]) * (events[i][:, 0] < support[0][1])
                valid_events_y = (
                    support[1][0] < events[i][:, 1]) * (events[i][:, 1] < support[1][1])

                valid_events_time = events[i][:, 2] < end_time
                valid_events = valid_events_x * valid_events_y * valid_events_time
                valid = events[i][valid_events]
                sorting = np.argsort(valid[:, 2])
                events[i] = valid[sorting]
            break

        it += 1

    return events, immigrants


def truncated_inv_power_law_custom(space_values, kernel_params):
    """Truncated inverse power law for the spatial kernel"""
    m, d = kernel_params

    q = -3/2

    values = torch.pow((1 + torch.square(space_values - m[0, 0])/d), q)

    return values

def truncated_gaussian_custom(space_values, kernel_params):
    """Truncated Gaussian for the spatial kernel"""
    m, sigma = kernel_params
    n_space_values = len(space_values)

    values = torch.zeros(n_space_values, n_space_values)

    values = torch.exp((- (torch.square(space_values - m[0, 0]))
                                    / (2 * torch.square(sigma[0, 0]))))
    
    return values

def truncated_gaussian_custom_time(time_values, kernel_params):
    """Truncated Gaussian for the temporal kernel"""
    m, sigma = kernel_params
    n_time_values = len(time_values)

    values = torch.zeros(n_time_values)

    values = torch.exp((- (torch.square(time_values - m[0, 0]))
                                    / (2 * torch.square(sigma[0, 0]))))
    
    return values


def truncated_exponential_custom(time_values, kernel_params):
    """Truncated exponential for the temporal kernel"""
    decay = kernel_params[0]

    values = decay * torch.exp(-decay * time_values)

    return values

def kumaraswamy_custom(time_values, kernel_params):
    """Kumaraswamy for the temporal kernel"""
    a, b = kernel_params

    values = torch.zeros(len(time_values))
    pa = a[0, 0] - 1
    pb = b[0, 0] - 1
    values = (a[0, 0] * b[0, 0] * (time_values**pa)
                    * ((1 - time_values**a[0, 0]) ** pb))

    return values

def raised_cosine_custom(time_values, kernel_params):
    """Raised cosine for the temporal kernel"""
    u, sigma = kernel_params

    values = torch.zeros(len(time_values))
    values = (1 + torch.cos(((time_values - u[0, 0]) / sigma[0, 0]
                                    * np.pi) - np.pi))

    return values