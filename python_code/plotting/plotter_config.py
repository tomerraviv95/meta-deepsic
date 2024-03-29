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
               'Joint ResNet10': 'blue',
               'Online DeepSIC': 'green',
               'Online ResNet10': 'green',
               'Meta-DeepSIC': 'red',
               'Meta-ResNet10': 'red',
               'Online DeepSIC - Modular Training': 'darkgreen',
               'Meta-DeepSIC - Modular Training': 'darkred'}

MARKERS_DICT = {'Joint DeepSIC': 'x',
                'Joint ResNet10': 'x',
                'Online DeepSIC': '.',
                'Online ResNet10': '.',
                'Meta-DeepSIC': 'd',
                'Meta-ResNet10': 'd',
                'Online DeepSIC - Modular Training': '.',
                'Meta-DeepSIC - Modular Training': 'd'}

LINESTYLES_DICT = {'Joint DeepSIC': '-',
                   'Joint ResNet10': 'dotted',
                   'Online DeepSIC': '-',
                   'Online ResNet10': 'dotted',
                   'Meta-DeepSIC': '-',
                   'Meta-ResNet10': 'dotted',
                   'Online DeepSIC - Modular Training': 'dashdot',
                   'Meta-DeepSIC - Modular Training': 'dashdot'}
