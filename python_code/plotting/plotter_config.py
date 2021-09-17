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
               'Joint DeepRX': 'blue',
               'Online DeepSIC': 'black',
               'Online DeepRX': 'black',
               'Meta-DeepSIC': 'green',
               'Meta-DeepRX': 'green',
               'Online DeepSIC - Single User': 'black',
               'Meta-DeepSIC - Single User': 'green'}

MARKERS_DICT = {'Joint DeepSIC': 'x',
                'Joint DeepRX': 'o',
                'Online DeepSIC': 'x',
                'Online DeepRX': 'o',
                'Meta-DeepSIC': 'x',
                'Meta-DeepRX': 'o',
                'Online DeepSIC - Single User': '+',
                'Meta-DeepSIC - Single User': '+'}

LINESTYLES_DICT = {'Joint DeepSIC': 'dotted',
                   'Joint DeepRX': 'dotted',
                   'Online DeepSIC': 'dotted',
                   'Online DeepRX': 'dotted',
                   'Meta-DeepSIC': 'dotted',
                   'Meta-DeepRX': 'dotted',
                   'Online DeepSIC - Single User': '-',
                   'Meta-DeepSIC - Single User': '-'}
