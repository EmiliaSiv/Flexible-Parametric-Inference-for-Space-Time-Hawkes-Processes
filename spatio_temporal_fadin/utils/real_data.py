import pandas as pd
import numpy as np

def read_real_data(file_name, end_time):
    """Collect the real data from a csv file.

    Parameters
    ----------
    file_name : `str`
        File name of the real data.
    
    end_time : `int`
        T the stopping time of the MSTH process

    Returns
    ----------
    n_dim : `int`
        D the dimension of the MSTH process
    
    events : list of arrays
        Events of the real data

    spatio_bound : list of shape (2,)
        Spatial bounds
    
    S : list of shape (2,)
        [S_X, S_Y] the spatial domain
    """
    df = pd.read_csv(file_name + ".csv", delim_whitespace=True, skiprows=0)
    data = df[['Date', 'Time', 'Lat', 'Lon']]
    data['temp'] = pd.to_datetime(data['Date'] + ' ' + data['Time']).copy()
    data = data.drop(columns=['Date', 'Time'])

    min_time = data['temp'].min()
    max_time = data['temp'].max()

    data['norm_temp'] = (data['temp'] - min_time).dt.total_seconds() / \
        (max_time - min_time).total_seconds() * end_time

    data = data.drop(columns=['temp'])

    events = [np.array(data)]
    n_dim = len(events)

    events_time = [events[i][:,2] for i in range(n_dim)]
    events_xcord = [events[i][:,0] for i in range(n_dim)]
    events_ycord = [events[i][:,1] for i in range(n_dim)]

    S1_x = int(np.floor(events_xcord[0].min()))
    S2_x = int(np.floor(events_xcord[0].max())+1)
    if (S2_x - S1_x)%2 == 0:
        S_x = [S1_x, S2_x]
    else:
        S2_x += 1
        S_x = [S1_x, S2_x]

    S1_y = int(np.floor(events_ycord[0].min()))
    S2_y = int(np.floor(events_ycord[0].max())+1)
    if (S2_y - S1_y)%2 == 0:
        S_y = [S1_y, S2_y]
    else:
        S2_y += 1
        S_y = [S1_y, S2_y]

    Kx = (S2_x - S1_x)/2
    Ky = (S2_y - S1_y)/2

    S = [Kx, Ky]

    spatio_bound = [S_x[0], S_y[0]]

    return n_dim, events, spatio_bound, S