import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

COLORS_DICT = {'Seq. DeepSIC, CSI uncertainty': 'pink',
               'Seq. DeepSIC, perfect CSI': 'pink'}

MARKERS_DICT = {'Seq. DeepSIC, CSI uncertainty': 'x',
                'Seq. DeepSIC, perfect CSI': 'o'}

LINESTYLES_DICT = {'Seq. DeepSIC, CSI uncertainty': 'dotted',
                   'Seq. DeepSIC, perfect CSI': 'solid'}

METHOD_NAMES = {'Seq. DeepSIC, CSI uncertainty': 'Seq. DeepSIC, CSI uncertainty',
                'Seq. DeepSIC, perfect CSI': 'Seq. DeepSIC, perfect CSI'}
