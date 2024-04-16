import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
plt.switch_backend('agg')
import seaborn as sns
import pandas as pd
import os
import pickle
from FileHelper import FileHelper
from ModelHelper import ModelHelper
import matplotlib.patches as patches
from matplotlib.patches import Circle


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
        - `density_vs_distance()`: Plot true and network density along lines of sight.	
        - `plot_model()`: Plot the model and save the plot.
        - `star_map()`: Plot the star map.
        - `density_true_vs_network()`: Plot true density vs network density.
        - `density_difference_vs_network()`: Plot the difference between the true and network density.
        
    # Example:
        The following example demonstrates how to use the `Visualizer` class to compare true and network extinction.
        
        >>> visualizer = Visualizer("config.json", dataset, 20.)
        >>> visualizer.compare_extinctions()
    """

    def __init__(self, config_file_path, dataset, max_distance):
        self.valdatas = None
        self.lossdatas = None
        self.ext_sight_datas = None
        self.dens_grid_datas = None
        self.ext_grid_datas = None
        self.dens_sight_datas = None
        self.config_file_path = config_file_path
        self.ext_grid_filename = FileHelper.give_config_value(self.config_file_path, "ext_grid_file")
        self.dens_grid_filename = FileHelper.give_config_value(self.config_file_path, "dens_grid_file")
        self.ext_los_filename = FileHelper.give_config_value(self.config_file_path, "ext_los_file")
        self.dens_los_filename = FileHelper.give_config_value(self.config_file_path, "dens_los_file")
        self.dataset = dataset
        self.max_distance = max_distance
        self.load_datas()
        
    def load_datas(self):
        """
        Load grid and line-of-sight data.
        
        # Remarks:
            This method loads the grid and line-of-sight data from the files specified in the configuration file.
            It checks if the files exist and loads the data if they do.
        """
        if os.path.exists(self.ext_grid_filename):
            self.ext_grid_datas = np.load(self.ext_grid_filename)
        if os.path.exists(self.dens_grid_filename):
            self.dens_grid_datas = np.load(self.dens_grid_filename)
        if os.path.exists(self.ext_los_filename):
            self.ext_sight_datas = np.load(self.ext_los_filename)
        if os.path.exists(self.dens_los_filename):
            self.dens_sight_datas = np.load(self.dens_los_filename)
        lossfile = FileHelper.give_config_value(self.config_file_path, "lossfile")
        valfile = FileHelper.give_config_value(self.config_file_path, "valfile")
        self.lossdatas = pd.read_csv(lossfile)
        self.valdatas = pd.read_csv(valfile)
        
    def plot_model(self):
        """
        Plot the model (the dataset file) and save the plot in the 'Plots' subdirectory of the current test directory.
        """
        file_model = FileHelper.give_config_value(self.config_file_path, "model_file")
        file_model_plot = FileHelper.give_config_value(self.config_file_path, "model_plot")
        with open(file_model, "rb") as file:
            model = pickle.load(file)
            file.close()
        
        x, y = np.mgrid[-5:5.1:0.1, -5:5.1:0.1]
        dens = x*0.
        for i in range(len(x[:, 1])):
            for j in range(len(x[1, :])):
                dens[i, j] = ModelHelper.compute_extinction_model_density(model, x[i, j], y[i, j], 0.)
                
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X (kpc)', fontsize=15)
        plt.ylabel('Y (kpc)', fontsize=15)
        plt.pcolormesh(x, y, dens, shading='auto', cmap="inferno")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(file_model_plot)
       
    def loss_function(self):
        """
        Plot the training and validation loss and save the plot in the 'Plots' subdirectory of the current test directory.
        """
        loss_plot_path = FileHelper.give_config_value(self.config_file_path, "loss_plot")
        sns.set_theme()
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sns.lineplot(data=self.lossdatas, x='Epoch', y='TotalLoss', label='Training loss', ax=ax)
        sns.lineplot(data=self.valdatas, x='Epoch', y='TotalValLoss', label='Validation loss', ax=ax)
        ax.set_xlabel('Epochs', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(loss_plot_path)
        # plt.show()

    def compare_densities(self):
        """
        Compare true and network density predictions and save the plot in the 'Plots' subdirectory of the current
        test directory.
        
        # Plot Structure:
            - True density map.
            - Network density map.
            - True-Network density map.
        """
        x = self.dens_grid_datas['X']
        y = self.dens_grid_datas['Y']
        dens_true = self.dens_grid_datas['density_model']
        dens_network = self.dens_grid_datas['density_network']
        density_plot_path = FileHelper.give_config_value(self.config_file_path, "density_plot")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(35, 10))
        divnorm = colors.TwoSlopeNorm(vcenter=0)
        palette = plt.cm.get_cmap("twilight")
        palette = palette.reversed()
        palette = self.truncate_colormap(palette, 0.1, 0.9)
        cs = ax1.set_title('True density', fontsize=15)
        cs1 = ax1.pcolormesh(x, y, dens_true, shading='auto', cmap=palette, norm=divnorm)
        cs = ax2.set_title('Network density', fontsize=15)
        cs2 = ax2.pcolormesh(x, y, dens_network * 2. / self.max_distance, shading='auto', cmap=palette, norm=divnorm)
        cs = ax3.set_title('True-Network', fontsize=15)
        cs3 = ax3.pcolormesh(x, y, (dens_true - dens_network * 2. / self.max_distance), shading='auto',
                             cmap='seismic', norm=divnorm)
        ax1.set_xlabel('X (kpc)', fontsize=15)
        ax1.set_ylabel('Y (kpc)', fontsize=15)
        ax2.set_xlabel('X (kpc)', fontsize=15)
        ax2.set_ylabel('Y (kpc)', fontsize=15)
        ax3.set_xlabel('X (kpc)', fontsize=15)
        ax3.set_ylabel('Y (kpc)', fontsize=15)
        cbar1 = fig.colorbar(cs1, ax=ax1)
        cbar2 = fig.colorbar(cs2, ax=ax2)
        cbar3 = fig.colorbar(cs3, ax=ax3)
        cbar1.ax.set_ylabel('Density (kpc$^{-2}$)', rotation=270, fontsize=15, labelpad=15)
        cbar2.ax.set_ylabel('Density (kpc$^{-2}$)', rotation=270, fontsize=15, labelpad=15)
        cbar3.ax.set_ylabel('Density (kpc$^{-2}$)', rotation=270, fontsize=15, labelpad=15)
        
        # Ajouter un cercle de rayon 5.5 centré en (0, 0)
        circle = plt.Circle((0, 0), 5.5, color='white', fill=False)
        circle1 = plt.Circle((0, 0), 5.5, color='white', fill=False)
        circle2 = plt.Circle((0, 0), 5.5, color='black', fill=False)
        ax1.add_artist(circle)
        ax2.add_artist(circle1)
        ax3.add_artist(circle2)
        
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        ax3.set_aspect('equal', adjustable='box')

        plt.savefig(density_plot_path)
        
    def model_histogram(self):
        x = self.dens_grid_datas['X']
        y = self.dens_grid_datas['Y']
        dens_true = self.dens_grid_datas['density_model']
        density_plot_path = FileHelper.give_config_value(self.config_file_path, "model_historgram_plot")

        df = pd.DataFrame({'X': x.flatten(), 'Y': y.flatten(), 'Density': dens_true.flatten()})

        # Crée un jointplot avec histogrammes marginaux
        g = sns.jointplot(data=df, x='X', y='Y', kind='hist', bins=(len(x), len(y)))

        g.ax_marg_y.cla()
        g.ax_marg_x.cla()
        
        norm = colors.TwoSlopeNorm(vcenter=0)
        
        sns.heatmap(data=dens_true.reshape(len(x), len(y)).T, ax=g.ax_joint, cbar=False, cmap='RdBu',xticklabels=10,
                    yticklabels=10, square=True, norm=norm)
        g.ax_joint.set_xticklabels(np.linspace(-5, 5, 11).astype(int), rotation=0)
        g.ax_joint.set_yticklabels(np.linspace(-5, 5, 11).astype(int), rotation=0)
        g.ax_joint.invert_yaxis()
        
        g.ax_marg_y.barh(np.arange(0.5, len(y)), np.sum(dens_true, axis=0), color='cornflowerblue')
        g.ax_marg_x.bar(np.arange(0.5, len(x)), np.sum(dens_true, axis=1), color='cornflowerblue')

        g.ax_marg_x.tick_params(axis='x', bottom=False, labelbottom=False)
        g.ax_marg_y.tick_params(axis='y', left=False, labelleft=False)
        g.ax_marg_x.tick_params(axis='y', left=False, labelleft=False)
        g.ax_marg_y.tick_params(axis='x', bottom=False, labelbottom=False)
        
        g.ax_joint.set_xlabel('X (kpc)', fontsize=15)
        g.ax_joint.set_ylabel('Y (kpc)', fontsize=15)
        g.ax_marg_x.set_ylabel('Density (kpc$^{-2}$)', fontsize=15)
        g.ax_marg_y.set_xlabel('Density (kpc$^{-2}$)', fontsize=15)

        plt.savefig(density_plot_path)
        
    def network_density_histogram(self):
        x = self.dens_grid_datas['X']
        y = self.dens_grid_datas['Y']
        dens_network = self.dens_grid_datas['density_network']
        density_plot_path = FileHelper.give_config_value(self.config_file_path, "network_density_histogram_plot")

        df = pd.DataFrame({'X': x.flatten(), 'Y': y.flatten(), 'Density': dens_network.flatten()})
        norm = colors.TwoSlopeNorm(vcenter=0)

        g = sns.jointplot(data=df, x='X', y='Y', kind='hist', bins=(len(x), len(y)))

        g.ax_marg_y.cla()
        g.ax_marg_x.cla()
        
        sns.heatmap(data=dens_network.reshape(len(x), len(y)).T, ax=g.ax_joint, cbar=False, cmap='RdBu',xticklabels=10,
                    yticklabels=10, square=True, norm=norm)
        g.ax_joint.set_xticklabels(np.linspace(-5, 5, 11).astype(int), rotation=0)
        g.ax_joint.set_yticklabels(np.linspace(-5, 5, 11).astype(int), rotation=0)
        g.ax_joint.invert_yaxis()
        
        g.ax_marg_y.barh(np.arange(0.5, len(y)), np.sum(dens_network, axis=0), color='cornflowerblue')
        g.ax_marg_x.bar(np.arange(0.5, len(x)), np.sum(dens_network, axis=1), color='cornflowerblue')

        g.ax_marg_x.tick_params(axis='x', bottom=False, labelbottom=False)
        g.ax_marg_y.tick_params(axis='y', left=False, labelleft=False)
        g.ax_marg_x.tick_params(axis='y', left=False, labelleft=False)
        g.ax_marg_y.tick_params(axis='x', bottom=False, labelbottom=False)
        
        g.ax_joint.set_xlabel('X (kpc)', fontsize=15)
        g.ax_joint.set_ylabel('Y (kpc)', fontsize=15)
        g.ax_marg_x.set_ylabel('Density (kpc$^{-2}$)', fontsize=15)
        g.ax_marg_y.set_xlabel('Density (kpc$^{-2}$)', fontsize=15)

        plt.savefig(density_plot_path)

    def compare_extinctions(self):
        """
        Compare true and network extinction predictions and save the plot in the Plots subdirectory
        of the current test directory.
        
        # Plot Structure:
            - True extinction map.
            - Network extinction map.
            - True-Network extinction map.
        """
        x = self.ext_grid_datas['X']
        y = self.ext_grid_datas['Y']
        ext_true = self.ext_grid_datas['extinction_model']
        ext_network = self.ext_grid_datas['extinction_network']
        extinction_plot_path = FileHelper.give_config_value(self.config_file_path, "extinction_plot")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
        palette = plt.cm.get_cmap("twilight")
        palette = self.truncate_colormap(palette, 0.0, 0.5)
        palette = palette.reversed()
        cs = ax1.set_title('True Extinction (mag)', fontsize=15)
        cs1 = ax1.pcolormesh(x, y, ext_true, shading='auto', cmap=palette, vmin=0.)
        cs = ax2.set_title('Network Extinction (mag)', fontsize=15)
        cs2 = ax2.pcolormesh(x, y, ext_network, shading='auto', cmap=palette, vmin=0.) 
        ax1.set_xlabel('X (kpc)', fontsize=15)
        ax1.set_ylabel('Y (kpc)', fontsize=15)
        ax2.set_xlabel('X (kpc)', fontsize=15)
        ax2.set_ylabel('Y (kpc)', fontsize=15)
        cbar1 = fig.colorbar(cs1, ax=ax1)
        cbar2 = fig.colorbar(cs2, ax=ax2)
        cbar1.ax.set_ylabel('Extinction (mag)', rotation=270, fontsize=15, labelpad=15)
        cbar2.ax.set_ylabel('Extinction (mag)', rotation=270, fontsize=15, labelpad=15)
        plt.savefig(extinction_plot_path)
        # plt.show()
        
    def extinction_vs_distance(self):
        """
        Plot true and network extinction along lines of sight and save the plot in the 'Plots' subdirectory of the
        
        # Plot Structure:
            The plot contains 8 subplots, each showing the true and network extinction along a line of sight.
        """
        ells = self.ext_sight_datas['ells']
        distance = self.ext_sight_datas['distance']
        los_ext_true = self.ext_sight_datas['los_ext_true']
        los_ext_network = self.ext_sight_datas['los_ext_network']
        extinction_los_plot_path = FileHelper.give_config_value(self.config_file_path, "extinction_los_plot")
        
        sns.set_theme()
        fig, axes = plt.subplots(2, 4, figsize=(35, 20))
        delta = 0.5
        
        for i in range(len(ells)):
            ax = axes[i//4, i % 4]
            ttl = 'l=' + str(ells[i])
            ax.set_title(ttl)
            sns.lineplot(x=distance, y=los_ext_true[i, :], ax=ax, label='True extinction')
            sns.lineplot(x=distance, y=los_ext_network[i, :], ax=ax, label='Network extinction')
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
            ax.set_xlabel('d (kpc)', fontsize=15)
            ax.set_ylabel('K (mag)', fontsize=15)
            
        plt.legend()
        plt.savefig(extinction_los_plot_path)
        
    def density_vs_distance(self):
        """
        Plot true and network density along lines of sight.
        
        # Plot Structure:
            The plot contains 8 subplots, each showing the true and network density along a line of sight.
        """
        ells = self.dens_sight_datas['ells']
        distance = self.dens_sight_datas['distance']
        los_dens_true = self.dens_sight_datas['los_dens_true']
        los_dens_network = self.dens_sight_datas['los_dens_network']
        density_los_plot_path = FileHelper.give_config_value(self.config_file_path, "density_los_plot")
        
        sns.set_theme()
        fig, axes = plt.subplots(2, 4, figsize=(35, 20))
        delta = 0.5
        
        for i in range(len(ells)):
            ax = axes[i//4, i % 4]
            ttl = 'l=' + str(ells[i])
            ax.set_title(ttl)
            
            sns.lineplot(x=distance, y=los_dens_true[i, :], ax=ax, label='True density')
            sns.lineplot(x=distance, y=los_dens_network[i, :], ax=ax, label='Network density')
            
            ax.set_xlabel('d (kpc)', fontsize=15)
            ax.set_ylabel('Density (kpc$^{-2}$)', fontsize=15)
            
        plt.legend()
        plt.savefig(density_los_plot_path)
        
    def star_map(self):
        distance = self.dataset.distance
        cosell = self.dataset.cosell
        sinell = self.dataset.sinell
        
        x = distance * cosell
        y = distance * sinell
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        
        sns.set_theme()
        sns.scatterplot(x=x, y=y, hue=self.dataset.K, palette='viridis', size=1, legend=False, alpha=0.8)
        
        # Ajouter un carré de 5 par 5 centré en (0, 0)
        ax = plt.gca()
        square = patches.Rectangle((-5, -5), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(square)
        ax.set_aspect('equal', adjustable='box')

        plt.xlabel('X (kpc)', fontsize=15)
        plt.ylabel('Y (kpc)', fontsize=15)
        
        plt.savefig("star_map.png")

    @staticmethod
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        '''
        https://stackoverflow.com/a/18926541
        '''
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
        
    def density_true_vs_network(self):
        """
        Plot true density vs network density.
        """
        dens_true = self.dens_grid_datas['density_model']
        dens_network = self.dens_grid_datas['density_network']
        density_plot_path = FileHelper.give_config_value(self.config_file_path, "density_true_vs_network_plot")
        
        X = self.dens_grid_datas['X']
        Y = self.dens_grid_datas['Y']
        distance = np.sqrt(X**2 + Y**2)
        
        mask = np.where(distance <= self.max_distance)
        
        dens_true = dens_true[mask]
        dens_network = dens_network[mask]

        line = np.linspace(0, max(dens_network.flatten()), 100)
        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.set_title('True density vs Network density', fontsize=15)
        sns.scatterplot(x = dens_network.flatten(), y = dens_true.flatten(), ax=ax)
        sns.lineplot(x=line, y=line, color='red', ax=ax)
        ax.set_xlabel('Network density (kpc$^{-2}$)', fontsize=15)
        ax.set_ylabel('True density (kpc$^{-2}$)', fontsize=15)
        plt.savefig(density_plot_path)
        
    def density_difference_vs_network(self):
        """
        Plot the difference between the true and network density in function of the network density.
        """
        dens_true = self.dens_grid_datas['density_model']
        dens_network = self.dens_grid_datas['density_network']
        density_plot_path = FileHelper.give_config_value(self.config_file_path, "density_difference_vs_network_plot")
        
        X = self.dens_grid_datas['X']
        Y = self.dens_grid_datas['Y']
        distance = np.sqrt(X**2 + Y**2)
        
        mask = np.where(distance <= self.max_distance)
        
        dens_true = dens_true[mask]
        dens_network = dens_network[mask]
        
        x_line = np.linspace(0, max(dens_true.flatten()), 100)
        sns.set_theme()
        _, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.set_title('Difference between True and Network density vs Network density', fontsize=15)
        sns.scatterplot(x = dens_true.flatten(), y = dens_true.flatten() - dens_network.flatten(), ax = ax)
        sns.lineplot(x = x_line, y = np.zeros(100), color = 'red', ax = ax)
        ax.set_xlabel('True density (kpc$^{-2}$)', fontsize=15)
        ax.set_ylabel('True density - Network density (kpc$^{-2}$)', fontsize=15)
        plt.savefig(density_plot_path)
        
    # 