import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from FileHelper import FileHelper
import os


class ModelVisualizer:
    """
    A class providing methods for visualizing extinction model predictions.
    
    # Args:
        - `config_file_name (str)`: Name of the configuration file.
        - `dataset (ExtinctionDataset)`: Instance of the extinction dataset.
        - `max_distance (float)`: Maximum distance in the dataset.
    
    # Attributes:
        - `config_file_name (str)`: Name of the configuration file.
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

    # Example:
        >>> # Instantiate a neural network model
        >>> your_model = YourExtinctionModel()
        >>> 
        >>> # Visualize the model predictions
        >>> ModelVisualizer.visualize_model(your_model)
    """

    def __init__(self, config_file_name, dataset, max_distance):
        self.valdatas = None
        self.lossdatas = None
        self.ext_sight_datas = None
        self.dens_grid_datas = None
        self.ext_grid_datas = None
        self.config_file_name = config_file_name
        self.ext_grid_filename = FileHelper.give_config_value(self.config_file_name, "ext_grid_file")
        self.dens_grid_filename = FileHelper.give_config_value(self.config_file_name, "dens_grid_file")
        self.ext_los_filename = FileHelper.give_config_value(self.config_file_name, "ext_los_file")
        self.dens_los_filename = FileHelper.give_config_value(self.config_file_name, "dens_los_file")
        self.dataset = dataset
        self.max_distance = max_distance
        self.load_datas()

    def loss_function(self):
        """
        Plot the training and validation loss and save the plot in the Plots subdirectory of the current test directory.
        """
        loss_plot_path = FileHelper.give_config_value(self.config_file_name, "loss_plot")
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
        # self.dens_sight_datas = np.load(self.dens_los_filename)
        lossfile = FileHelper.give_config_value(self.config_file_name, "lossfile")
        valfile = FileHelper.give_config_value(self.config_file_name, "valfile")
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
        density_plot_path = FileHelper.give_config_value(self.config_file_name, "density_plot")

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
        extinction_plot_path = FileHelper.give_config_value(self.config_file_name, "extinction_plot")

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
        """
        Plot true and network extinction along lines of sight and save the plot in the Plots subdirectory of
        the current test directory.
        """
        ells = self.ext_sight_datas['ells']
        distance = self.ext_sight_datas['distance']
        los_ext_true = self.ext_sight_datas['los_ext_true']
        los_ext_network = self.ext_sight_datas['los_ext_network']
        extinction_los_plot_path = FileHelper.give_config_value(self.config_file_name, "extinction_los_plot")

        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(35, 20))
        delta = 0.5
        ttl = 'l=' + str(ells[0])
        ax1.set_title(ttl)
        ax1.plot(distance, los_ext_true[0, :], label='True extinction')
        ax1.plot(distance, los_ext_network[0, :], label='Network extinction')
        xdata = []
        ydata = []
        errdata = []
        for i in range(self.dataset.__len__()):
            if ells[0] - delta < self.dataset.ell[i].item() <= ells[0] + delta:
                xdata.append((self.dataset.distance[i].item()))
                ydata.append(self.dataset.K[i])
                errdata.append(self.dataset.error[i].item())
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        recerr1 = distance * 0.
        for i in range(len(distance)):
            idx = np.where(np.abs(distance[i] - xdata) < 0.2)
            if len(idx[0]) > 0:
                recerr1[i] = np.var(los_ext_network[0, i] - ydata[idx])
            else:
                recerr1[i] = los_ext_network[0, i] * los_ext_network[0, i]
        ax1.errorbar(xdata, ydata, yerr=errdata, fmt='o')
        ax1.set_xlabel('d (kpc)')
        ax1.set_ylabel('K (mag)')

        ttl = 'l=' + str(ells[1])
        ax2.set_title(ttl)
        ax2.plot(distance, los_ext_true[1, :], label='True extinction')
        ax2.plot(distance, los_ext_network[1, :], label='Network extinction')
        xdata = []
        ydata = []
        errdata = []
        for i in range(self.dataset.__len__()):
            if ells[1] - delta < self.dataset.ell[i].item() <= ells[1] + delta:
                xdata.append((1. + self.dataset.distance[i].item()) * self.max_distance / 2.)
                ydata.append(self.dataset.K[i])
                errdata.append(self.dataset.error[i].item())
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        recerr2 = distance * 0.
        for i in range(len(distance)):
            idx = np.where(abs(distance[i] - xdata) < 0.2)
            if len(idx[0]) > 0:
                recerr2[i] = np.var(los_ext_network[1, i] - ydata[idx])
            else:
                recerr2[i] = los_ext_network[1, i] * los_ext_network[1, i]
        ax2.errorbar(xdata, ydata, yerr=errdata, fmt='o')
        ax2.set_xlabel('d (kpc)')
        ax2.set_ylabel('K (mag)')

        ttl = 'l=' + str(ells[2])
        ax3.set_title(ttl)
        ax3.plot(distance, los_ext_true[2, :], label='True extinction')
        ax3.plot(distance, los_ext_network[2, :], label='Network extinction')
        xdata = []
        ydata = []
        errdata = []
        for i in range(self.dataset.__len__()):
            if ells[2] - delta < self.dataset.ell[i].item() <= ells[2] + delta:
                xdata.append((1. + self.dataset.distance[i].item()) * self.max_distance / 2.)
                ydata.append(self.dataset.K[i])
                errdata.append(self.dataset.error[i].item())
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        recerr3 = distance * 0.
        for i in range(len(distance)):
            idx = np.where(abs(distance[i] - xdata) < 0.2)
            if len(idx[0]) > 0:
                recerr3[i] = np.var(los_ext_network[2, i] - ydata[idx])
            else:
                recerr3[i] = los_ext_network[2, i] * los_ext_network[2, i]
        ax3.errorbar(xdata, ydata, yerr=errdata, fmt='o')
        ax3.set_xlabel('d (kpc)')
        ax3.set_ylabel('K (mag)')

        ttl = 'l=' + str(ells[3])
        ax4.set_title(ttl)
        ax4.plot(distance, los_ext_true[3, :], label='True extinction')
        ax4.plot(distance, los_ext_network[3, :], label='Network extinction')
        xdata = []
        ydata = []
        errdata = []
        for i in range(self.dataset.__len__()):
            if ells[3] - delta < self.dataset.ell[i].item() <= ells[3] + delta:
                xdata.append((1. + self.dataset.distance[i].item()) * self.max_distance / 2.)
                ydata.append(self.dataset.K[i])
                errdata.append(self.dataset.error[i].item())
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        recerr4 = distance * 0.
        for i in range(len(distance)):
            idx = np.where(abs(distance[i] - xdata) < 0.2)
            if len(idx[0]) > 0:
                recerr4[i] = np.var(los_ext_network[3, i] - ydata[idx])
            else:
                recerr4[i] = los_ext_network[3, i] * los_ext_network[3, i]
        ax4.errorbar(xdata, ydata, yerr=errdata, fmt='o')
        ax4.set_xlabel('d (kpc)')
        ax4.set_ylabel('K (mag)')

        ttl = 'l=' + str(ells[4])
        ax5.set_title(ttl)
        ax5.plot(distance, los_ext_true[4, :], label='True extinction')
        ax5.plot(distance, los_ext_network[4, :], label='Network extinction')
        xdata = []
        ydata = []
        errdata = []
        for i in range(self.dataset.__len__()):
            if ells[4] - delta < self.dataset.ell[i].item() <= ells[4] + delta:
                xdata.append((1. + self.dataset.distance[i].item()) * self.max_distance / 2.)
                ydata.append(self.dataset.K[i])
                errdata.append(self.dataset.error[i].item())
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        recerr5 = distance * 0.
        for i in range(len(distance)):
            idx = np.where(abs(distance[i] - xdata) < 0.2)
            if len(idx[0]) > 0:
                recerr5[i] = np.var(los_ext_network[4, i] - ydata[idx])
            else:
                recerr5[i] = los_ext_network[4, i] * los_ext_network[4, i]
        ax5.errorbar(xdata, ydata, yerr=errdata, fmt='o')
        ax5.set_xlabel('d (kpc)')
        ax5.set_ylabel('K (mag)')

        ttl = 'l=' + str(ells[5])
        ax6.set_title(ttl)
        ax6.plot(distance, los_ext_true[5, :], label='True extinction')
        ax6.plot(distance, los_ext_network[5, :], label='Network extinction')
        xdata = []
        ydata = []
        errdata = []
        for i in range(self.dataset.__len__()):
            if ells[5] - delta < self.dataset.ell[i].item() <= ells[5] + delta:
                xdata.append((1. + self.dataset.distance[i].item()) * self.max_distance / 2.)
                ydata.append(self.dataset.K[i])
                errdata.append(self.dataset.error[i].item())
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        recerr6 = distance * 0.
        for i in range(len(distance)):
            idx = np.where(abs(distance[i] - xdata) < 0.2)
            if len(idx[0]) > 0:
                recerr6[i] = np.var(los_ext_network[5, i] - ydata[idx])
            else:
                recerr6[i] = los_ext_network[5, i] * los_ext_network[5, i]
        ax6.errorbar(xdata, ydata, yerr=errdata, fmt='o')
        ax6.set_xlabel('d (kpc)')
        ax6.set_ylabel('K (mag)')

        ttl = 'l=' + str(ells[6])
        ax7.set_title(ttl)
        ax7.plot(distance, los_ext_true[6, :], label='True extinction')
        ax7.plot(distance, los_ext_network[6, :], label='Network extinction')
        xdata = []
        ydata = []
        errdata = []
        for i in range(self.dataset.__len__()):
            if ells[6] - delta < self.dataset.ell[i].item() <= ells[6] + delta:
                xdata.append((1. + self.dataset.distance[i].item()) * self.max_distance / 2.)
                ydata.append(self.dataset.K[i])
                errdata.append(self.dataset.error[i].item())
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        recerr7 = distance * 0.
        for i in range(len(distance)):
            idx = np.where(abs(distance[i] - xdata) < 0.2)
            if len(idx[0]) > 0:
                recerr7[i] = np.var(los_ext_network[6, i] - ydata[idx])
            else:
                recerr7[i] = los_ext_network[6, i] * los_ext_network[6, i]
        ax7.errorbar(xdata, ydata, yerr=errdata, fmt='o')
        ax7.set_xlabel('d (kpc)')
        ax7.set_ylabel('K (mag)')

        ttl = 'l=' + str(ells[7])
        ax8.set_title(ttl)
        ax8.plot(distance, los_ext_true[7, :], label='True extinction')
        ax8.plot(distance, los_ext_network[7, :], label='Network extinction')
        xdata = []
        ydata = []
        errdata = []
        for i in range(self.dataset.__len__()):
            if ells[7] - delta < self.dataset.ell[i].item() <= ells[7] + delta:
                xdata.append((1. + self.dataset.distance[i].item()) * self.max_distance / 2.)
                ydata.append(self.dataset.K[i])
                errdata.append(self.dataset.error[i].item())
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        recerr8 = distance * 0.
        for i in range(len(distance)):
            idx = np.where(abs(distance[i] - xdata) < 0.2)
            if len(idx[0]) > 0:
                recerr8[i] = np.var(los_ext_network[7, i] - ydata[idx])
            else:
                recerr8[i] = los_ext_network[7, i] * los_ext_network[7, i]
        ax8.errorbar(xdata, ydata, yerr=errdata, fmt='o')
        ax8.set_xlabel('d (kpc)')
        ax8.set_ylabel('K (mag)')

        plt.legend()
        plt.savefig(extinction_los_plot_path)
        # plt.show()
