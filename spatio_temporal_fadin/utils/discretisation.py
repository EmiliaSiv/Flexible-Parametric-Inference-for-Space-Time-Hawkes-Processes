import torch
import numpy as np

def smooth_projection_spatio_temp(events, n_grid_spatio, n_grid_time,
                                  delta_spatio, delta_time, spatio_bound):
    """Project events on the grid.

    Parameters
        ----------
        events : list of array
            The timestamps of the spatial point process' events.

        n_grid_spatio : `array`, shape (2,)
            [G_X, G_Y] the number of points on each component of the discretized
            spatial grid.
        
        n_grid_time : `int`
            G_T the number of points on the discretized temporal grid.

        delta_spatio: 'array', shape (2,)
            [Delta_X, Delta_Y] the stepsizes of each component of the discretized
            spatial grid.
        
        delta_time: 'int'
            Delta_T the stepsize of the discretized temporal grid.

        spatio_bound: 'array', shape (2,)
            [K_X, K_Y] the spatial bound of each component of the spatial domain.

        Returns
        ----------
        events_grid : `tensor`, shape (n_dim, n_grid_spatio[0], n_grid_spatio[1], \
            n_grid_time)
            Events projected on the pre-defined grid.
        
        events_smooth : `array`, shape (n_dim, n_events, 3)
            Events smoothed.
        
        coord_grid: 'array', shape (n_dim, n_events, 3)
            Coordinates projected on the pre-defined grid.
    """
    bounds_x = spatio_bound[0]
    bounds_y = spatio_bound[1]
    n_dim = len(events)
    events_grid = np.zeros((n_dim, n_grid_spatio[0], n_grid_spatio[1], n_grid_time))

    events_xcord = [events[i][:, 0] for i in range(n_dim)]
    events_ycord = [events[i][:, 1] for i in range(n_dim)]
    events_time = [events[i][:, 2] for i in range(n_dim)]

    events_smooth = [None] * n_dim
    coord_grid = [None] * n_dim
    for j in range(n_dim):
        ei_xcord = events_xcord[j]
        ei_ycord = events_ycord[j]
        ei_time = events_time[j]

        temp_xcord = np.round((ei_xcord-bounds_x) / delta_spatio[0]).astype(np.int64)
        temp_ycord = np.round((ei_ycord-bounds_y) / delta_spatio[1]).astype(np.int64)
        temp_time = np.round(ei_time / delta_time).astype(np.int64)

        coord_grid[j] = [temp_xcord, temp_ycord, temp_time]

        cat_coord = np.concatenate((temp_xcord[:, None],
                                    temp_ycord[:, None],
                                    temp_time[:, None]), axis=1)
        indices, count = np.unique(cat_coord, return_counts=True, axis=0)
        events_grid[j, indices[:, 0], indices[:, 1], indices[:, 2]] += count

        temp2_xcord = bounds_x + temp_xcord * delta_spatio[0]
        temp2_ycord = bounds_y + temp_ycord * delta_spatio[1]
        temp2_time = temp_time * delta_time
        events_smooth[j] = np.concatenate((temp2_xcord[:, None],
                                           temp2_ycord[:, None],
                                           temp2_time[:, None]), axis=1)

    return events_grid, events_smooth, coord_grid