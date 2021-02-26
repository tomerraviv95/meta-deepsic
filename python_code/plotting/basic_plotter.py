import matplotlib.pyplot as plt
from dataclasses import *
from matplotlib import rc


fig1 = plt.figure(figsize=(6, 5))
ax = fig1.gca()

plt.semilogy(conf.snr_list, BERs, linestyle='--', marker='d', color='red', label=r'DeepSIC - Sequential - Perfect CSI',
             linewidth=2.5, fillstyle='none', markersize=10, markeredgewidth=2)

plt.title(r'DeepSIC: BPSK, $6\times 6$ MIMO System')
plt.ylabel(r'$\textbf{BER}$', fontsize=20)
plt.xlabel(r'SNR [dB]', fontsize=20)
plt.legend(handlelength=5)
plt.xticks(conf.snr_list)
plt.grid(True, which='both')
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

# fig1.savefig('BER_DeepSIC.pdf', dpi=60)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
plt.rc('font', weight='bold')