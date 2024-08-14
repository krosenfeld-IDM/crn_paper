import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

dist = sps.weibull_min(c=2.5, scale=3)

mean, var, skew, kurt = dist.stats(moments='mvsk')
print(mean)


x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
ax.plot(x, dist.pdf(x), 'r-', lw=5, alpha=0.6, label='dist')
plt.axvline(x=mean, ls=':')

plt.show()