import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


loss = pd.read_csv('loss.csv',header=None)
val = pd.read_csv('val.csv',header=None)

plt.figure()
#plot first and second column of loss dataframes
plt.plot(loss[0],loss[1],label="Loss train")
#plot first and second column of val dataframes
plt.plot(val[0],val[1],'--',label="Loss validation")

plt.yscale('log')
plt.legend(loc="upper right")
plt.show()
