import numpy as np

def precomp_phi_grid(n_support_spatio, n_support_time, events_grid, n_events):
    """Compute the precomputation constant phi summed on the grid.

    Parameters
    ----------
    n_support_spatio : `array`, shape (2,)
        [L_X, L_Y] the number of points on each component of the discretized
        spatial support.
    
    n_support_time : `int`
        L_T the number of points on the discretized temporal support.

    events_grid : `tensor`, shape (n_dim, n_grid_spatio[0], n_grid_spatio[1], \
        n_grid_time)
        Events projected on the pre-defined grid.

    n_events : `array`, shape (n_dim,)
        The number of events for each process.

    Returns
    ----------
    phi_grid : `array`, shape (n_dim, n_support_spatio[0], n_support_spatio[1], \
        n_support_time)
        The precomputation constant phi.
    """
    n_dim = len(n_events)
    st_pos_x = n_support_spatio[0] // 2
    st_pos_y = n_support_spatio[1] // 2
    phi_grid = np.zeros((n_dim, n_support_spatio[0], n_support_spatio[1], \
                         n_support_time))
    for i in range(n_dim):
        ev_i_neg = events_grid[i]
        ev_i = np.flip(events_grid[i])
        n_ev_i = n_events[i]
        phi_grid[i, :, :, :] = n_ev_i

        # Varying one dimension
        temp_time = np.cumsum(ev_i[:, :, :n_support_time], axis=2).sum((0, 1))
        const_time = np.roll(temp_time, 1)
        const_time[0] = 0.

        temp_x = np.cumsum(ev_i[:st_pos_x+1, :, :], axis=0).sum((1, 2))
        temp_x_neg = np.cumsum(ev_i_neg[:st_pos_x, :, :], axis=0).sum((1, 2))
        temp_x_neg = np.flip(temp_x_neg)
        const_x = np.roll(temp_x, 1)
        const_x[0] = 0.
        cx = np.concatenate((temp_x_neg, const_x))

        temp_y = np.cumsum(ev_i[:, :st_pos_y+1, :], axis=1).sum((0, 2))
        temp_y_neg = np.cumsum(ev_i_neg[:, :st_pos_y, :], axis=1).sum((0, 2))
        temp_y_neg = np.flip(temp_y_neg)
        const_y = np.roll(temp_y, 1)
        const_y[0] = 0.
        cy = np.concatenate((temp_y_neg, const_y))

        # Varying two dimension
        shifts_pos_pos = ev_i[:st_pos_x+1, :st_pos_y+1]
        shifts_pos_neg = ev_i[:st_pos_x+1, -st_pos_y:]
        shifts_neg_pos = ev_i[-st_pos_x:, :st_pos_y+1]
        shifts_neg_neg = ev_i_neg[:st_pos_x, :st_pos_y]

        cumsum_pp = np.cumsum(np.cumsum(shifts_pos_pos, axis=0), axis=1).sum(2)
        roll_pp = np.roll(np.roll(cumsum_pp, 1, axis=0), 1, axis=1)

        cumsum_pn = np.cumsum(
            np.cumsum(np.flip(shifts_pos_neg, axis=1), axis=0), axis=1).sum(2)
        roll_pn = np.roll(cumsum_pn, 1, axis=0)
        roll_pn_flip = np.flip(roll_pn, axis=1)

        cumsum_np = np.cumsum(
            np.cumsum(np.flip(shifts_neg_pos, axis=0), axis=0), axis=1).sum(2)
        roll_np = np.roll(cumsum_np, 1,  axis=1)
        roll_np_flip = np.flip(roll_np, axis=0)

        cumsum_nn = np.cumsum(np.cumsum(shifts_neg_neg, axis=0), axis=1).sum(2)
        cumsum_nn_flip = np.flip(cumsum_nn)

        top_matrix = np.concatenate((cumsum_nn_flip, roll_np_flip), axis=1)
        bottom_matrix = np.concatenate((roll_pn_flip, roll_pp), axis=1)
        cxy = np.concatenate((top_matrix, bottom_matrix), axis=0)
        cxy[st_pos_x, :] = 0.
        cxy[:, st_pos_y] = 0.

        temp_xt = np.cumsum(
            np.cumsum(ev_i[:st_pos_x+1, :, :n_support_time], axis=0), axis=2).sum(1)
        const_xt = np.roll(np.roll(temp_xt, 1, axis=0), 1, axis=1)

        temp_xt_neg = np.cumsum(
            np.cumsum(ev_i_neg[:st_pos_x, :, :n_support_time], axis=0), axis=2).sum(1)
        temp_xt_neg = np.roll(temp_xt_neg, 1, axis=1)

        temp_xt_neg = np.flip(temp_xt_neg, axis=0)
        cxt = np.concatenate((temp_xt_neg, const_xt))
        cxt[st_pos_x, :] = 0.
        cxt[:, 0] = 0.

        temp_yt = np.cumsum(
            np.cumsum(ev_i[:, :st_pos_y+1, :n_support_time], axis=1), axis=2).sum(0)
        const_yt = np.roll(np.roll(temp_yt, 1, axis=0), 1, axis=1)

        temp_yt_neg = np.cumsum(
            np.cumsum(
                ev_i_neg[:, :st_pos_y, :n_support_time], axis=1), axis=2).sum(0)
        temp_yt_neg = np.roll(temp_yt_neg, 1, axis=1)

        temp_yt_neg = np.flip(temp_yt_neg, axis=0)
        cyt = np.concatenate((temp_yt_neg, const_yt))
        cyt[st_pos_y, :] = 0.
        cyt[:, 0] = 0.

        # Varying three dimension

        shifts_pos_pos_pos = ev_i[:st_pos_x+1, :st_pos_y+1, :n_support_time]
        shifts_pos_neg_pos = ev_i[:st_pos_x+1, -st_pos_y:, :n_support_time]
        shifts_neg_pos_pos = ev_i[-st_pos_x:, :st_pos_y+1, :n_support_time]
        shifts_neg_neg_pos = ev_i_neg[:st_pos_x, :st_pos_y, :n_support_time]

        cumsum_ppp = np.cumsum(
            np.cumsum(np.cumsum(shifts_pos_pos_pos, axis=0), axis=1), axis=2)
        roll_ppp = np.roll(
            np.roll(np.roll(cumsum_ppp, 1, axis=0), 1, axis=1), 1, axis=2)
        cumsum_pnp = np.cumsum(
            np.cumsum(
                np.cumsum(np.flip(shifts_pos_neg_pos, axis=1), axis=0), axis=1), axis=2)
        roll_pnp = np.roll(np.roll(cumsum_pnp, 1, axis=0), 1, axis=2)
        roll_pnp_flip = np.flip(roll_pnp, axis=1)

        cumsum_npp = np.cumsum(
            np.cumsum(
                np.cumsum(np.flip(shifts_neg_pos_pos, axis=0), axis=0), axis=1), axis=2)
        roll_npp = np.roll(np.roll(cumsum_npp, 1,  axis=1), 1, axis=2)
        roll_npp_flip = np.flip(roll_npp, axis=0)

        cumsum_nnp = np.cumsum(
            np.cumsum(np.cumsum(shifts_neg_neg_pos, axis=0), axis=1), axis=2)
        roll_nnp = np.roll(cumsum_nnp, 1, axis=2)
        roll_nnp_flip = np.flip(roll_nnp, axis=(0, 1))

        top_matrix = np.concatenate((roll_nnp_flip, roll_npp_flip), axis=1)
        bottom_matrix = np.concatenate((roll_pnp_flip, roll_ppp), axis=1)
        cxyt = np.concatenate((top_matrix, bottom_matrix), axis=0)
        cxyt[st_pos_x, :, :] = 0.
        cxyt[:, st_pos_y, :] = 0.
        cxyt[:, :, 0] = 0.

        phi_grid[i] += cxy[:, :, None] + cxt[:, None, :] + \
            cyt[None, :, :] - cx[:, None, None] - cy[None, :, None] \
            - const_time[None, None, :] - cxyt

    return phi_grid

def precomp_phi_events_prepsi(coord_grid, kernel_length, events_smooth,
                         n_support_spatio, n_support_time, n_events):
    
    """Compute the precomputation constant phi summed on the projected events, \
        and a revised version used for the approximation term of the precomputation psi.

    Parameters
    ----------
    coord_grid: 'array', shape (n_dim, n_events, 3)
        Coordinates projected on the pre-defined grid.

    kernel_length: 'array', shape (2,)
        Spatial (of shape (2,)) and temporal kernel lengths.

    events_smooth : 'array', shape (n_dim, n_events, 3)
        Events smoothed.

    n_support_spatio : 'array', shape (2,)
        [L_X, L_Y] the number of points on each component of the discretized
        spatial support.
    
    n_support_time : 'int'
        L_T the number of points on the discretized temporal support.

    n_events : 'array', shape (n_dim,)
        The number of events for each process.

    Returns
    ----------
    result_prepsi : `array`, shape (n_dim, n_dim, 2*n_support_spatio[0]-1, \
        2*n_support_spatio[1]-1, n_support_time)
        The pre-version of the approximation of the constant psi.

    result_phi : `array`, shape (n_dim, n_dim, n_support_spatio[0], \
        n_support_spatio[1], n_support_time)
        The precomputation constant phi.
    """
    W_s, W_t = kernel_length
    tol = 1e-6

    start_positive_x = n_support_spatio[0] // 2
    start_positive_y = n_support_spatio[1] // 2
    result_prepsi = np.zeros((1, 1, 2*n_support_spatio[0] - 1, 2*n_support_spatio[1] - 1, n_support_time))
    result_phi = np.zeros((1, 1, n_support_spatio[0], n_support_spatio[1], n_support_time))

    ev_i = events_smooth[0]

    count_ij_prepsi = []
    count_ij_phi = []
    old_k = 0

    for ix, ei in enumerate(ev_i):
        # Getting the starting indices in events[j] for the events ei.
        k = old_k
        while ev_i[k, 2] < ei[2]:
            k += 1
            if k >= n_events[0] - 1:
                break
        c = k
        old_k = np.maximum(0, k-1)
        events_in_support_prepsi = []
        events_in_support_phi = []
        if ev_i[c, 2] >= ei[2] - tol:
            while ev_i[c, 2] <= ei[2] + W_t + tol:
                if ev_i[c, 0] >= ei[0] - 2*W_s[0] - tol and ev_i[c, 0] <= ei[0] + 2*W_s[0] + tol \
                    and ev_i[c, 1] >= ei[1] - 2*W_s[1] - tol and \
                        ev_i[c, 1] <= ei[1] + 2*W_s[1] + tol:

                    events_in_support_prepsi.append(np.array([coord_grid[0][0][c],
                                                       coord_grid[0][1][c],
                                                       coord_grid[0][2][c]]))
                    
                # Getting the events in the spatio-temporal support
                if ev_i[c, 0] >= ei[0] - W_s[0] - tol and ev_i[c, 0] <= ei[0] + W_s[0] + tol \
                    and ev_i[c, 1] >= ei[1] - W_s[1] - tol and \
                        ev_i[c, 1] <= ei[1] + W_s[1] + tol:
                    
                    events_in_support_phi.append(np.array([coord_grid[0][0][c],
                                                       coord_grid[0][1][c],
                                                       coord_grid[0][2][c]]))
                    
                c += 1
                if c >= n_events[0]:
                    break

            if len(events_in_support_prepsi) > 0:
                events_in_support_prepsi -= np.array([coord_grid[0][0][ix],
                                               coord_grid[0][1][ix],
                                               coord_grid[0][2][ix]])

                events_in_support_prepsi[:, 0] += 2*start_positive_x
                events_in_support_prepsi[:, 1] += 2*start_positive_y

                count_ij_prepsi.append(events_in_support_prepsi.astype(np.int64))

            if len(events_in_support_phi) > 0:
                events_in_support_phi -= np.array([coord_grid[0][0][ix],
                                               coord_grid[0][1][ix],
                                               coord_grid[0][2][ix]])

                events_in_support_phi[:, 0] += start_positive_x
                events_in_support_phi[:, 1] += start_positive_y

                count_ij_phi.append(events_in_support_phi.astype(np.int64))

    # Getting the delay on the grid of these events
    if len(count_ij_prepsi) > 0:
        count_ij_flatten = np.vstack(count_ij_prepsi)
        a, b = np.unique(count_ij_flatten, return_counts=True, axis=0)
        for k in range(len(a)):
            result_prepsi[0, 0, a[k][0],  a[k][1],  a[k][2]] += b[k]

    if len(count_ij_phi) > 0:
        count_ij_flatten = np.vstack(count_ij_phi)
        a, b = np.unique(count_ij_flatten, return_counts=True, axis=0)
        for k in range(len(a)):
            result_phi[0, 0, a[k][0],  a[k][1],  a[k][2]] += b[k]

    return result_prepsi, result_phi