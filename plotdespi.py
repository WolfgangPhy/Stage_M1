import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


loss = np.loadtxt('fiducial_model2D_mean.loss')
val = np.loadtxt('fiducial_model2D_mean.val')

plt.figure()
plt.plot(loss[:,0],loss[:,1],label="Loss train")
plt.plot(val[:,0],val[:,1],'--',label="Loss validation")

plt.yscale('log')
plt.legend(loc="upper right")
plt.show()
