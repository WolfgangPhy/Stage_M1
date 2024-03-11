import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle
from FileHelper import FileHelper
from ModelHelper import ModelHelper


class Visualizer:
    """
    A class providing methods for visualizing extinction model predictions.
    
    # Args:
        - `config_file_path (str)`: Name of the configuration file.
        - `dataset (ExtinctionDataset)`: Instance of the extinction dataset.
        - `max_distance (float)`: Maximum distance in the dataset.
    
    # Attributes:
        - `config_file_path (str)`: Name of the configuration file.
        - `ext_grid_filename (str)`: Filename for the extinction grid data.
        - `dens_grid_filename (str)`: Filename for the density grid data.
        - `ext_los_filename (str)`: Filename for the extinction line-of-sight data.
        - `dens_los_filename (str)`: Filename for the density line-of-sight data.
        - `dataset (ExtinctionDataset)`: Instance of the extinction dataset.
        - `max_distance (float)`: Maximum distance in the dataset.
        - `ext_grid_datas (dict)`: Dictionary containing extinction grid data.
        - `dens_grid_datas (dict)`: Dictionary containing density grid data.
        - `ext_sight_datas (dict)`: Dictionary containing extinction line-of-sight data.
        - `dens_sight_datas (dict)`: Dictionary containing density line-of-sight data.
        - `lossdatas (pd.DataFrame)`: DataFrame containing training loss data.
        - `valdatas (pd.DataFrame)`: DataFrame containing validation loss data.
        
    
    # Methods:
        - `loss_function()`: Plot the training and validation loss.
        - `load_datas()`: Load grid and line-of-sight data.
        - `compare_densities()`: Compare true and network density predictions.
        - `compare_extinctions()`: Compare true and network extinction predictions.
        - `extinction_vs_distance()`: Plot true and network extinction along lines of sight.
    """

    def __init__(self, config_file_path, dataset, max_distance):
        self.valdatas = None
        self.lossdatas = None
        self.ext_sight_datas = None
        self.dens_grid_datas = None
        self.ext_grid_datas = None
        self.config_file_path = config_file_path
        self.ext_grid_filename = FileHelper.give_config_value(self.config_file_path, "ext_grid_file")
        self.dens_grid_filename = FileHelper.give_config_value(self.config_file_path, "dens_grid_file")
        self.ext_los_filename = FileHelper.give_config_value(self.config_file_path, "ext_los_file")
        self.dens_los_filename = FileHelper.give_config_value(self.config_file_path, "dens_los_file")
        self.dataset = dataset
        self.max_distance = max_distance
        self.load_datas()

    def loss_function(self):
        """
        Plot the training and validation loss and save the plot in the Plots subdirectory of the current test directory.
        """
        loss_plot_path = FileHelper.give_config_value(self.config_file_path, "loss_plot")
        sns.set_theme()
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sns.lineplot(data=self.lossdatas, x='Epoch', y='TotalLoss', label='Training loss', ax=ax)
        sns.lineplot(data=self.valdatas, x='Epoch', y='TotalValLoss', label='Validation loss', ax=ax)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(loss_plot_path)
        # plt.show()

    def load_datas(self):
        """
        Load grid and line-of-sight data.
        """
        if os.path.exists(self.ext_grid_filename):
            self.ext_grid_datas = np.load(self.ext_grid_filename)
        if os.path.exists(self.dens_grid_filename):
            self.dens_grid_datas = np.load(self.dens_grid_filename)
        if os.path.exists(self.ext_los_filename):
            self.ext_sight_datas = np.load(self.ext_los_filename)
        #self.dens_sight_datas = np.load(self.dens_los_filename)
        lossfile = FileHelper.give_config_value(self.config_file_path, "lossfile")
        valfile = FileHelper.give_config_value(self.config_file_path, "valfile")
        self.lossdatas = pd.read_csv(lossfile)
        self.valdatas = pd.read_csv(valfile)

    def compare_densities(self):
        """
        Compare true and network density predictions and save the plot in the Plots subdirectory of the current
        test directory.
        """
        x = self.dens_grid_datas['X']
        y = self.dens_grid_datas['Y']
        dens_true = self.dens_grid_datas['density_model']
        dens_network = self.dens_grid_datas['density_network']
        density_plot_path = FileHelper.give_config_value(self.config_file_path, "density_plot")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(35, 10))
        cs = ax1.set_title('True density')
        cs1 = ax1.pcolormesh(x, y, dens_true, shading='auto', vmin=0., vmax=30, cmap="inferno")
        cs = ax2.set_title('Network density')
        cs2 = ax2.pcolormesh(x, y, dens_network * 2. / self.max_distance, shading='auto', vmin=0., vmax=30,
                             cmap="inferno")
        cs = ax3.set_title('True-Network (%)')
        cs3 = ax3.pcolormesh(x, y, (dens_true - dens_network * 2. / self.max_distance), vmin=-2, vmax=2, shading='auto',
                             cmap="inferno")
        ax1.set_xlabel('X (kpc)')
        ax1.set_ylabel('Y (kpc)')
        ax2.set_xlabel('X (kpc)')
        ax2.set_ylabel('Y (kpc)')
        ax3.set_xlabel('X (kpc)')
        ax3.set_ylabel('Y (kpc)')
        fig.colorbar(cs1, ax=ax1)
        fig.colorbar(cs2, ax=ax2)
        fig.colorbar(cs3, ax=ax3)
        plt.savefig(density_plot_path)
        # plt.show()

    def compare_extinctions(self):
        """
        Compare true and network extinction predictions and save the plot in the Plots subdirectory
        of the current test directory.
        """
        x = self.ext_grid_datas['X']
        y = self.ext_grid_datas['Y']
        ext_true = self.ext_grid_datas['extinction_model']
        ext_network = self.ext_grid_datas['extinction_network']
        extinction_plot_path = FileHelper.give_config_value(self.config_file_path, "extinction_plot")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(35, 10))
        cs = ax1.set_title('True Extinction')
        cs1 = ax1.pcolormesh(x, y, ext_true, shading='auto', cmap="inferno")
        cs = ax2.set_title('Network Extinction')
        cs2 = ax2.pcolormesh(x, y, ext_network, shading='auto', cmap="inferno")
        cs = ax3.set_title('True-Network (%)')
        cs3 = ax3.pcolormesh(x, y, np.abs(ext_true - ext_network) / ext_true * 100., vmin=0, vmax=50, shading='auto',
                             cmap="inferno")
        ax1.set_xlabel('X (kpc)')
        ax1.set_ylabel('Y (kpc)')
        ax2.set_xlabel('X (kpc)')
        ax2.set_ylabel('Y (kpc)')
        ax3.set_xlabel('X (kpc)')
        ax3.set_ylabel('Y (kpc)')
        fig.colorbar(cs1, ax=ax1)
        fig.colorbar(cs2, ax=ax2)
        fig.colorbar(cs3, ax=ax3)
        plt.savefig(extinction_plot_path)
        # plt.show()
        
    def extinction_vs_distance(self):
        ells = self.ext_sight_datas['ells']
        distance = self.ext_sight_datas['distance']
        los_ext_true = self.ext_sight_datas['los_ext_true']
        los_ext_network = self.ext_sight_datas['los_ext_network']
        extinction_los_plot_path = FileHelper.give_config_value(self.config_file_path, "extinction_los_plot")
        
        fig, axes = plt.subplots(2, 4, figsize=(35, 20))
        delta = 0.5
        
        for i in range(len(ells)):
            ax = axes[i//4, i%4]
            ttl = 'l=' + str(ells[i])
            ax.set_title(ttl)
            ax.plot(distance, los_ext_true[i, :], label='True extinction')
            ax.plot(distance, los_ext_network[i, :], label='Network extinction')
            xdata = []
            ydata = []
            errdata = []
            for j in range(self.dataset.__len__()):
                if ells[i] - delta < self.dataset.ell[j].item() <= ells[i] + delta:
                    xdata.append(self.dataset.distance[j].item())
                    ydata.append(self.dataset.K[j])
                    errdata.append(self.dataset.error[j].item())
            xdata = np.array(xdata)
            ydata = np.array(ydata)
            recerr = distance * 0.
            for j in range(len(distance)):
                idx = np.where(abs(distance[j] - xdata) < 0.2)
                if len(idx[0]) > 0:
                    recerr[j] = np.var(los_ext_network[i, j] - ydata[idx])
                else:
                    recerr[j] = los_ext_network[i, j] * los_ext_network[i, j]
            ax.errorbar(xdata, ydata, yerr=errdata, fmt='o')
            ax.set_xlabel('d (kpc)')
            ax.set_ylabel('K (mag)')
            
        plt.legend()
        plt.savefig(extinction_los_plot_path)
        
    def plot_model(self):
        file_model = FileHelper.give_config_value(self.config_file_path, "model_file")
        file_model_plot = FileHelper.give_config_value(self.config_file_path, "model_plot")
        with open(file_model,"rb") as file:
            model = pickle.load(file)
            file.close()
        
        X,Y = np.mgrid[-5:5.1:0.1, -5:5.1:0.1]
        dens = X*0.
        for i in range(len(X[:,1])):
            for j in range(len(X[1,:])):
                dens[i,j] = ModelHelper.compute_extinction_model_density(model, X[i, j], Y[i, j], 0.)
                
        plt.pcolormesh(X, Y, dens, shading='auto', cmap="inferno")
        plt.savefig(file_model_plot)
        
