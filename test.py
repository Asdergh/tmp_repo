import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import cwt
from scipy.signal import ricker



t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) * np.exp(t**2 + 1) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.ricker, widths)

plt.style.use("dark_background")
fig, axis = plt.subplots()
axis.imshow(cwtmatr, cmap="jet")

plt.show()
