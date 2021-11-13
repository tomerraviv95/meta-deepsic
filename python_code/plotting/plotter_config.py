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

COLORS_DICT = {'Joint DeepSIC': 'blue',
               'Joint BlackBox': 'blue',
               'Online DeepSIC': 'green',
               'Online BlackBox': 'green',
               'Meta-DeepSIC': 'red',
               'Meta-BlackBox': 'red',
               'Online DeepSIC - Single User': 'darkgreen',
               'Meta-DeepSIC - Single User': 'darkred'}

MARKERS_DICT = {'Joint DeepSIC': 'x',
                'Joint BlackBox': 'x',
                'Online DeepSIC': '.',
                'Online BlackBox': '.',
                'Meta-DeepSIC': 'd',
                'Meta-BlackBox': 'd',
                'Online DeepSIC - Single User': '.',
                'Meta-DeepSIC - Single User': 'd'}

LINESTYLES_DICT = {'Joint DeepSIC': '-',
                   'Joint BlackBox': 'dotted',
                   'Online DeepSIC': '-',
                   'Online BlackBox': 'dotted',
                   'Meta-DeepSIC': '-',
                   'Meta-BlackBox': 'dotted',
                   'Online DeepSIC - Single User': 'dashdot',
                   'Meta-DeepSIC - Single User': 'dashdot'}
