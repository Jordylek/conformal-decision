import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import seaborn as sns




plt.rc('figure', dpi=120)  # was 250
xsmallfontsize = 7
smallfontsize = 10
fontsize = 15
bigfontsize = 20
xbigfontsize = 25
xxbigfontsize = 30
# fontsize = 20
smallfigsize = (8, 6)
figsize = (12, 8)
subplot_figsize = (14, 6)
bigfigsize = (16, 8)  # necessary for plot with histogram
# plt.rcParams["figure.figsize"] = figsize
mpl.rc('xtick', labelsize=bigfontsize)
mpl.rc('ytick', labelsize=bigfontsize)
plt.rcParams.update(**{'figure.figsize': smallfigsize, 'lines.linewidth': 3, 'lines.markersize': 3,
                       'legend.fontsize': fontsize, 'axes.grid': False, 'grid.alpha': 0.4,
                       'xtick.labelsize': xbigfontsize, 'ytick.labelsize': xbigfontsize, 'legend.framealpha': 0.5,
                       'figure.dpi': 120, 'hist.bins': 20, 'axes.labelsize': bigfontsize,
                       'axes.titlesize': xbigfontsize , 'font.serif': 'Times New Roman', 'font.family': "Times New Roman",
                       'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def save_figure(fig, filepath):
    if not filepath.endswith('.pdf'):
        filepath = filepath + '.pdf'
    fig.savefig(filepath, bbox_inches='tight', format='pdf')