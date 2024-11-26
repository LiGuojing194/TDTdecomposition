import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='serif')

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid(ls=':')

plt.savefig('./output/plot.pdf', format='pdf', dpi=900, bbox_inches='tight')


