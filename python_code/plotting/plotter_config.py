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
               'Online DeepSIC': 'black',
               'Meta-DeepSIC': 'green',
               'Online DeepSIC - Single User Training': 'black',
               'Meta-DeepSIC - Single User Training': 'green'}

MARKERS_DICT = {'Joint DeepSIC': 'x',
                'Online DeepSIC': 'o',
                'Meta-DeepSIC': '+',
                'Online DeepSIC - Single User Training': 'o',
                'Meta-DeepSIC - Single User Training': '+'}

LINESTYLES_DICT = {'Joint DeepSIC': 'dotted',
                   'Online DeepSIC': 'dotted',
                   'Meta-DeepSIC': 'dotted',
                   'Online DeepSIC - Single User Training': '-',
                   'Meta-DeepSIC - Single User Training': '-'}
