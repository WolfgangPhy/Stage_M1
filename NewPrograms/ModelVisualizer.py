import numpy as np
import matplotlib.pyplot as plt
import ExtinctionModelHelper as Helper

class ModelVisualizer:
    @staticmethod
    def visualize_model(model):
        X, Y = np.mgrid[-5:5.1:0.1, -5:5.1:0.1]
        dens = np.zeros_like(X)

        for i in range(len(X[:, 1])):
            for j in range(len(X[1, :])):
                dens[i, j] = Helper.ExtinctionModelHelper.compute_extinction_model_density(model, X[i, j], Y[i, j], 0.)

        plt.pcolormesh(X, Y, dens, shading='auto', cmap=plt.cm.gist_yarg)
        plt.show()