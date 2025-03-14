# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# from tueplots.bundles.iclr2023()
FONTSIZE = 20
plt.rcParams["figure.figsize"] = (5, 3.2)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.grid.axis"] = "y"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('legend', fontsize=FONTSIZE - 1)


def plot_fig2_paper(kernel='TG', leg=False):
    """
    kernel : str
        'TG'
    """

    df = pd.read_csv(f'results/exp2_{kernel}.csv')

    K = df["K"].unique()
    K.sort()
    dt = df["dt"].unique()
    dt.sort()

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

    fig, ax = plt.subplots(1, 1, figsize = (6,4))
    
    ind_color = 0
    for i, d in enumerate(dt):
        for j, k in enumerate(K):
            this_df = df.query("dt == @d and K == @k")
            curve = this_df.groupby("T")["time"].quantile(
                [0.25, 0.5, 0.75]).unstack()
            plt.loglog(
                curve.index, curve[0.5], "s-", lw=4, c=palette[ind_color],
                markersize=10, markevery=1
            )
            plt.fill_between(
                curve.index, curve[0.25], curve[0.75], alpha=0.2,
                color=palette[ind_color], hatch=None
            )
            ind_color += 1
            plt.xlim(1e1, 1e3)

    # Create legend

    custom_lines_T = [
        Line2D([], [], color=palette[i], lw=3) for i in range(4)
    ]
    if leg:
        plt.legend(
            custom_lines_T,
            [r"$%d$" % k for k in [10, 20]],
            title="$S$", loc="lower center",
            bbox_to_anchor=(0, 1, 1, 0.01), ncol=4
        )

    h = plt.xlabel(r'$T$')
    h.set_fontsize(26)
    c = 'k'
    h = plt.ylabel(r'Time (s.)', fontdict={'color': c})
    h.set_fontsize(26)


    plt.savefig(f"plots/fig2_{kernel}_comptime.png", bbox_inches='tight')
    plt.savefig(f"plots/fig2_{kernel}_comptime.pdf", bbox_inches='tight')


plt.close('all')

plot_fig2_paper(kernel='TG')