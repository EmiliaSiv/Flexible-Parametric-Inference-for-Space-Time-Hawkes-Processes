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


    

def plot_fig1_paper(kernel='TG', param='baseline', leg=False):
    """
    kernel : str
        'TG' | 'EXP' | 'KUR' | 'POW_KUR'
    """

    df = pd.read_csv(f'results/exp1_{kernel}.csv')

    T = df["T"].unique()
    T.sort()
    K = df["K"].unique()
    K.sort()

    new_col = 'l2{}'.format(param)
    col_param = 'param_{}'.format(param)

    palette = [matplotlib.cm.viridis_r(x) for x in np.linspace(0, 1, 5)][1:]

    fig, ax = plt.subplots(1, 1, figsize = (6,4))

    ind_color = 0
    for i, t in enumerate(T):
        for j, k in enumerate(K):
            this_df = df.query("T == @t and K == @k")
            this_df[new_col] = (this_df[col_param] - this_df[param])**2
            curve = this_df.groupby("dt")[new_col].quantile(
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
            ax.set_xlim(0.5, 0.05)

    ax.set_xticks([])       
    ax.set_xticks([0.05, 0.5], [0.05, 0.5], minor=True)

    custom_lines_T = [
        Line2D([], [], color=palette[i], lw=3) for i in range(4)
    ]
    if leg:
        plt.legend(
            custom_lines_T,
            [r"$%d$, $%d$" % (t, k) for t in [10, 100] for k in [10, 20]],
            title="$T$ and $S$", loc="lower center",
            bbox_to_anchor=(0, 1, 1, 0.01), ncol=4
        )

    
    h = plt.xlabel(r'$\Delta$')
    h.set_fontsize(26)
    c = 'k'
    h = plt.ylabel(r'$\ell_2$ error', fontdict={'color': c})
    h.set_fontsize(26)

    plt.savefig(f"plots/fig1_{kernel}_{param}.png", bbox_inches='tight')
    plt.savefig(f"plots/fig1_{kernel}_{param}.pdf", bbox_inches='tight')




plt.close('all')

plot_fig1_paper(kernel='TG', param='baseline')
plot_fig1_paper(kernel='TG', param='alpha')
plot_fig1_paper(kernel='TG', param='mean')
plot_fig1_paper(kernel='TG', param='sigma')
plot_fig1_paper(kernel='TG', param='mean_time')
plot_fig1_paper(kernel='TG', param='sigma_time')



plot_fig1_paper(kernel='EXP', param='baseline')
plot_fig1_paper(kernel='EXP', param='alpha')
plot_fig1_paper(kernel='EXP', param='mean')
plot_fig1_paper(kernel='EXP', param='sigma')
plot_fig1_paper(kernel='EXP', param='decay')



plot_fig1_paper(kernel='KUR', param='baseline')
plot_fig1_paper(kernel='KUR', param='alpha')
plot_fig1_paper(kernel='KUR', param='mean')
plot_fig1_paper(kernel='KUR', param='sigma')
plot_fig1_paper(kernel='KUR', param='a')
plot_fig1_paper(kernel='KUR', param='b')



plot_fig1_paper(kernel='POW_KUR', param='baseline')
plot_fig1_paper(kernel='POW_KUR', param='alpha')
plot_fig1_paper(kernel='POW_KUR', param='mean')
plot_fig1_paper(kernel='POW_KUR', param='sigma')
plot_fig1_paper(kernel='POW_KUR', param='a')
plot_fig1_paper(kernel='POW_KUR', param='b')