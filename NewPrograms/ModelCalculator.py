import numpy as np
import ExtinctionModelHelper as Helper
import ExtinctionNeuralNet as Net
import FileHelper as FHelper
import torch
from tqdm import tqdm
class ModelCalculator:
    """
    A class for computing density and extinction models on a grid and along lines of sight.

    # Args:
        - `model`: Extinction model for computation.
        - `builder`: Integral builder for computing network predictions.
        - `x_max`: Maximum x-coordinate for the grid.
        - `x_min`: Minimum x-coordinate for the grid.
        - `y_max`: Maximum y-coordinate for the grid.
        - `y_min`: Minimum y-coordinate for the grid.
        - `step`: Step size for the grid.
        - `max_distance`: Maximum distance for normalization.
        - `device`: Device (CPU/GPU) for PyTorch operations.
        - `network`: Neural network for predictions.
        - `config_file_path`: Path to the current test configuration file.

    # Attributes:
        - `model`: Extinction model for computation.
        - `builder`: Integral builder for computing network predictions.
        - `x_max`: Maximum x-coordinate for the grid.
        - `x_min`: Minimum x-coordinate for the grid.
        - `y_max`: Maximum y-coordinate for the grid.
        - `y_min`: Minimum y-coordinate for the grid.
        - `step`: Step size for the grid.
        - `max_distance`: Maximum distance for normalization.
        - `device`: Device (CPU/GPU) for PyTorch operations.
        - `network`: Neural network for predictions.
        - `config_file_path`: Path to the current test configuration file.
        - `ext_grid_filename`: Filename for the extinction grid file.
        - `ext_los_filename`: Filename for the extinction lines of sight file.
        - `dens_grid_filename`: Filename for the density grid file.
        - `dens_los_filename`: Filename for the density lines of sight file.

    # Methods:
        - `compute_extinction_grid()`: Compute extinction on a 2D grid.
        - `compute_density_grid()`: Compute density on a 2D grid.
        - `compute_extinction_sight()`: Compute extinction along lines of sight.
        - `compute_density_sight()`: Compute density along lines of sight.

    # Example:
        >>> # Instantiate a model calculator
        >>> model_calculator = ModelCalculator(model, builder, x_max, x_min, y_max, y_min, step, max_distance, device, network, config_file_path)
        >>> 
        >>> # Compute density and extinction on a 2D grid
        >>> model_calculator.density_extinction_grid()
        >>> 
        >>> # Compute density and extinction along lines of sight
        >>> model_calculator.density_extinction_sight()
    """
    def __init__(self, model, builder, x_max, x_min, y_max, y_min, step, max_distance, device, network, config_file_path):
        self.model = model
        self.builder = builder
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.step = step
        self.max_distance = max_distance
        self.device = device
        self.network = network
        self.config_file_path = config_file_path
        self.ext_grid_filename = FHelper.FileHelper.give_config_value(self.config_file_path, "ext_grid_file")
        self.ext_los_filename = FHelper.FileHelper.give_config_value(self.config_file_path, "ext_los_file")
        self.dens_grid_filename = FHelper.FileHelper.give_config_value(self.config_file_path, "dens_grid_file")
        self.dens_los_filename = FHelper.FileHelper.give_config_value(self.config_file_path, "dens_los_file")
        
    
    def compute_extinction_grid(self):
        """
        Compute extinction on a 2D grid.
        Save the results in a NumPy file in the NpzFiles subdirectory of the current test directory.
        """
        
        X, Y = np.mgrid[self.x_min:self.x_max:self.step, self.y_min:self.y_max:self.step]
        
        extinction_model = np.zeros_like(X)
        extinction_network = np.zeros_like(X)
        ell = np.zeros_like(X)
        R_model = np.zeros_like(X)
        R_network = np.sqrt(X*X + Y*Y)
        cosell = X/R_network
        sinell = Y/R_network
        
        print("Computing extinction on a 2D grid")
        for i in tqdm(range(len(X[:,1])), desc='Rows'):
            for j in tqdm(range(len(X[1,:])), desc=f'Column number {i}', leave=False):
                ell[i,j], R_model[i,j] = Helper.ExtinctionModelHelper.convert_cartesian_to_galactic_2D(X[i,j], Y[i,j])
                extinction_model[i,j] = Helper.ExtinctionModelHelper.integ_d(Helper.ExtinctionModelHelper.compute_extinction_model_density, ell[i,j], 0., R_model[i,j], self.model)
                
                data = torch.Tensor([cosell[i,j], sinell[i,j], 2.*R_network[i,j]/self.max_distance-1.]).float()
                data = data.unsqueeze(1)
                extinction_network[i,j] = self.builder.integral(torch.transpose(data.to(self.device), 0, 1), self.network, min_distance = -1.)
                
        np.savez(self.ext_grid_filename, extinction_model = extinction_model, extinction_network = extinction_network, X = X, Y = Y)
               
    def compute_density_grid(self):
        """
        Compute density on a 2D grid.
        Save the results in a NumPy file in the NpzFiles subdirectory of the current test directory.
        """
        
        X, Y = np.mgrid[self.x_min:self.x_max:self.step, self.y_min:self.y_max:self.step]
        
        density_model = np.zeros_like(X)
        density_network = np.zeros_like(X)
        R_network = np.sqrt(X*X + Y*Y)
        cosell = X/R_network
        sinell = Y/R_network
        
        print("Computing density on a 2D grid")
        for i in tqdm(range(len(X[:,1])), desc='Rows'):
            for j in tqdm(range(len(X[1,:])), desc=f'Column number {i}', leave=False):
                density_model[i,j] = Helper.ExtinctionModelHelper.compute_extinction_model_density(self.model, X[i,j], Y[i,j], 0.)
                
                data = torch.Tensor([cosell[i,j], sinell[i,j], 2.*R_network[i,j]/self.max_distance-1.]).float()
                
                density_network[i,j] = self.network.forward(data.to(self.device))
                
        np.savez(self.dens_grid_filename, density_model = density_model, density_network = density_network, X = X, Y = Y)
    
    def compute_extinction_sight(self):
        """
        Compute extinction along lines of sight.
        Save the results in the NpzFiles subdirectory of the current test directory.
        """
        ells = np.arange(0., 360., 45.)
        cosell = np.cos(ells*np.pi/180.)
        sinell = np.sin(ells*np.pi/180.)
        distance = np.linspace(0.,7., num = 71)
        
        los_ext_network = np.zeros((8,71))
        los_ext_true = np.zeros((8,71))

        print("Computing extinction along lines of sight")
        
        for i in tqdm(range(len(ells)), desc='Lines of sight'):
            for j in tqdm(range(len(distance)), desc=f'Distance number {i}', leave=False):
                data = torch.Tensor([cosell[i],sinell[i],2.*distance[j]/self.max_distance-1.]).float()
                data = data.unsqueeze(1)
                los_ext_network[i,j] = self.builder.integral(torch.transpose(data.to(self.device),0,1), self.network, min_distance=-1.)

                X = distance[j]*cosell[i]
                Y = distance[j]*sinell[i]
                los_ext_true[i,j] = Helper.ExtinctionModelHelper.integ_d(Helper.ExtinctionModelHelper.compute_extinction_model_density,ells[i],0.,distance[j], self.model)
                
        np.savez(self.ext_los_filename, ells = ells, distance = distance, los_ext_true = los_ext_true, los_ext_network = los_ext_network)
        
    def compute_density_sight(self):
        """
        Compute density along lines of sight.
        Save the results in the NpzFiles subdirectory of the current test directory.
        """
        ells = np.arange(0., 360., 45.)
        cosell = np.cos(ells*np.pi/180.)
        sinell = np.sin(ells*np.pi/180.)
        distance = np.linspace(0.,7., num = 71)

        los_dens_network = np.zeros((8,71))
        los_dens_true = np.zeros((8,71))

        print("Computing density along lines of sight")
        
        for i in tqdm(range(len(ells)), desc='Lines of sight'):
            for j in tqdm(range(len(distance)), desc=f'Distance number {i}', leave=False):
                data = torch.Tensor([cosell[i],sinell[i],2.*distance[j]/self.max_distance-1.]).float()
                los_dens_network[i,j] = self.network.forward(data.to(self.device))
                X = distance[j]*cosell[i]
                Y = distance[j]*sinell[i]
                los_dens_true[i,j] = Helper.ExtinctionModelHelper.compute_extinction_model_density(self.model, X,Y,0.)
                
        np.savez(self.dens_los_filename, ells = ells, distance = distance, los_dens_true = los_dens_true, los_dens_network = los_dens_network)