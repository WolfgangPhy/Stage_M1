import numpy as np
import matplotlib.pyplot as plt
import ExtinctionModelHelper as Helper

class ModelVisualizer:
    """
    A class providing static methods for visualizing extinction model predictions.

    # Methods:
        - `visualize_model(model)`: Visualizes the extinction model predictions.

    # Example:
        >>> # Instantiate a neural network model
        >>> your_model = YourExtinctionModel()
        >>> 
        >>> # Visualize the model predictions
        >>> ModelVisualizer.visualize_model(your_model)
    """
    @staticmethod
    def visualize_model(model):
        """
        Visualize the extinction model predictions.

        This method generates a 2D visualization of the extinction model predictions.
        It creates a meshgrid of points in the X-Y plane, computes the corresponding density values using the provided model,
        and plots the density values as a color mesh.

        # Args:
            - `model`: An extinction model for visualization.

        # Example:
            >>> # Instantiate a neural network model
            >>> your_model = YourExtinctionModel()
            >>> 
            >>> # Visualize the model predictions
            >>> ModelVisualizer.visualize_model(your_model)
        """
        X, Y = np.mgrid[-5:5.1:0.1, -5:5.1:0.1]
        dens = np.zeros_like(X)

        for i in range(len(X[:, 1])):
            for j in range(len(X[1, :])):
                dens[i, j] = Helper.ExtinctionModelHelper.compute_extinction_model_density(model, X[i, j], Y[i, j], 0.)

        plt.pcolormesh(X, Y, dens, shading='auto', cmap=plt.cm.gist_yarg)
        plt.show()