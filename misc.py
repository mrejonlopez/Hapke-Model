import numpy as np
import hapke
import matplotlib.pyplot as plt
from scipy import optimize

T = 100

n = hapke.opticalconstants(T)['n']
k = hapke.opticalconstants(T)['k']
wav = hapke.opticalconstants(T)['wav']

n2 = hapke.opticalconstants(T, crystallinity=False)['n']
k2 = hapke.opticalconstants(T, crystallinity=False)['k']
wav2 = hapke.opticalconstants(T, crystallinity=False)['wav']

int_opt = hapke.inter_optical_constants(wav, wav2, n, k)

wav = int_opt['wav']
n = int_opt['n']
k = int_opt['k']

phi, D, b = [0.35, 10 ** (-5), np.deg2rad(15)]
eme, inc, phase = [np.deg2rad(40), np.deg2rad(30), np.deg2rad(70)]
parameters = [phi, D, b]
angles = [eme, inc, phase]

b = [0.1, 0.25, 0.5, 0.75, 1]



fig, ax = plt.subplots()
for i in range(len(b)):
    ax.plot(wav, hapke.hapke_model([phi,D,b[i]], wav, angles, n, k)['IF'], label='b=' + str(b[i]))
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('I/F')
ax.set_title(f'')
ax.legend()
plt.show()
