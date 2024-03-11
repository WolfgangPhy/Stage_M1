import numpy as np
import torch
from tqdm import tqdm
from ModelHelper import ModelHelper
from NetworkHelper import NetworkHelper
from FileHelper import FileHelper


class Calculator:
    """
    A class for computing density and extinction of the model and the network on a grid and along lines of sight.

    # Args:
        - `model (ExtinctionModel)`: Extinction model for computation.
        - `x_max (int)`: Maximum x-coordinate for the grid.
        - `x_min (int)`: Minimum x-coordinate for the grid.
        - `y_max (int)`: Maximum y-coordinate for the grid.
        - `y_min (int)`: Minimum y-coordinate for the grid.
        - `step (int)`: Step size for the grid.
        - `max_distance (float)`: Maximum distance for normalization.
        - `device (torch.device)`: Device (CPU/GPU) for PyTorch operations.
        - `network (torch.nn.Module)`: Neural network for predictions.
        - `config_file_path (str)`: Path to the current test configuration file.

    # Attributes:
        - `model (ExtinctionModel)`: Extinction model for computation.
        - `x_max (int)`: Maximum x-coordinate for the grid.
        - `x_min (int)`: Minimum x-coordinate for the grid.
        - `y_max (int)`: Maximum y-coordinate for the grid.
        - `y_min (int)`: Minimum y-coordinate for the grid.
        - `step (int)`: Step size for the grid.
        - `max_distance (float)`: Maximum distance for normalization.
        - `device (torch.device)`: Device (CPU/GPU) for PyTorch operations.
        - `network (torch.nn.Module)`: Neural network for predictions.
        - `config_file_path (str)`: Path to the current test configuration file.
        - `ext_grid_filename (str)`: Filename for the extinction grid file.
        - `ext_los_filename (str)`: Filename for the extinction lines of sight file.
        - `dens_grid_filename (str)`: Filename for the density grid file.
        - `dens_los_filename (str)`: Filename for the density lines of sight file.

    # Methods:
        - `compute_extinction_grid()`: Compute extinction on a 2D grid.
        - `compute_density_grid()`: Compute density on a 2D grid.
        - `compute_extinction_sight()`: Compute extinction along lines of sight.
        - `compute_density_sight()`: Compute density along lines of sight.
    """
    def __init__(self, model, x_max, x_min, y_max, y_min, step, max_distance, device, network, config_file_path):
        self.model = model
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.step = step
        self.max_distance = max_distance
        self.device = device
        self.network = network
        self.config_file_path = config_file_path
        self.ext_grid_filename = FileHelper.give_config_value(self.config_file_path, "ext_grid_file")
        self.ext_los_filename = FileHelper.give_config_value(self.config_file_path, "ext_los_file")
        self.dens_grid_filename = FileHelper.give_config_value(self.config_file_path, "dens_grid_file")
        self.dens_los_filename = FileHelper.give_config_value(self.config_file_path, "dens_los_file")

    def compute_extinction_grid(self):
        """
        Compute extinction on a 2D grid.
        Save the results in a NumPy file in the NpzFiles subdirectory of the current test directory.
        """
        
        x, y = np.mgrid[self.x_min:self.x_max:self.step, self.y_min:self.y_max:self.step]
        
        extinction_model = np.zeros_like(x)
        extinction_network = np.zeros_like(x)
        ell = np.zeros_like(x)
        r_model = np.zeros_like(x)
        r_network = np.sqrt(x*x + y*y)
        cosell = x/r_network
        sinell = y/r_network
        
        print("Computing extinction on a 2D grid")
        for i in tqdm(range(len(x[:, 1])), desc='Rows'):
            for j in tqdm(range(len(x[1, :])), desc=f'Column number {i}', leave=False):
                ell[i, j], r_model[i, j] = (ModelHelper.
                                            convert_cartesian_to_galactic_2D(x[i, j], y[i, j])
                                            )
                extinction_model[i, j] = ModelHelper.integ_d(ModelHelper
                                                                       .compute_extinction_model_density,
                                                                       ell[i, j], 0., r_model[i, j], self.model
                                                                       )
                
                data = torch.Tensor([cosell[i, j], sinell[i, j], 2.*r_network[i, j]/self.max_distance-1.]).float()
                data = data.unsqueeze(1)
                extinction_network[i, j] = NetworkHelper.integral(torch.transpose(data.to(self.device), 0, 1),
                                                                 self.network, min_distance=-1.
                                                                 )
                
        np.savez(self.ext_grid_filename, extinction_model=extinction_model, extinction_network=extinction_network,
                 X=x, Y=y
                 )
               
    def compute_density_grid(self):
        """
        Compute density on a 2D grid.
        Save the results in a NumPy file in the NpzFiles subdirectory of the current test directory.
        """
        
        x, y = np.mgrid[self.x_min:self.x_max:self.step, self.y_min:self.y_max:self.step]
        
        density_model = np.zeros_like(x)
        density_network = np.zeros_like(x)
        r_network = np.sqrt(x*x + y*y)
        cosell = x/r_network
        sinell = y/r_network
        
        print("Computing density on a 2D grid")
        for i in tqdm(range(len(x[:, 1])), desc='Rows'):
            for j in tqdm(range(len(x[1, :])), desc=f'Column number {i}', leave=False):
                density_model[i, j] = ModelHelper.compute_extinction_model_density(self.model, x[i, j],
                                                                                             y[i, j], 0.)
                
                data = torch.Tensor([cosell[i, j], sinell[i, j], 2.*r_network[i, j]/self.max_distance-1.]).float()
                
                density_network[i, j] = self.network.forward(data.to(self.device))
                
        np.savez(self.dens_grid_filename, density_model=density_model, density_network=density_network, X=x, Y=y)
    
    def compute_extinction_sight(self):
        """
        Compute extinction along lines of sight.
        Save the results in the NpzFiles subdirectory of the current test directory.
        """
        ells = np.arange(0., 360., 45.)
        cosell = np.cos(ells*np.pi/180.)
        sinell = np.sin(ells*np.pi/180.)
        distance = np.linspace(0., 7., num=71)
        
        los_ext_network = np.zeros((8, 71))
        los_ext_true = np.zeros((8, 71))

        print("Computing extinction along lines of sight")
        
        for i in tqdm(range(len(ells)), desc='Lines of sight'):
            for j in tqdm(range(len(distance)), desc=f'Distance number {i}', leave=False):
                data = torch.Tensor([cosell[i], sinell[i], 2.*distance[j]/self.max_distance-1.]).float()
                data = data.unsqueeze(1)
                los_ext_network[i, j] = NetworkHelper.integral(torch.transpose(data.to(self.device), 0, 1),
                                                              self.network, min_distance=-1.
                                                              )

                los_ext_true[i, j] = ModelHelper.integ_d(ModelHelper
                                                                   .compute_extinction_model_density, ells[i],
                                                                   0., distance[j], self.model
                                                                   )
                
        np.savez(self.ext_los_filename, ells=ells, distance=distance, los_ext_true=los_ext_true,
                 los_ext_network=los_ext_network
                 )
        
    def compute_density_sight(self):
        """
        Compute density along lines of sight.
        Save the results in the NpzFiles subdirectory of the current test directory.
        """
        ells = np.arange(0., 360., 45.)
        cosell = np.cos(ells*np.pi/180.)
        sinell = np.sin(ells*np.pi/180.)
        distance = np.linspace(0., 7., num=71)

        los_dens_network = np.zeros((8, 71))
        los_dens_true = np.zeros((8, 71))

        print("Computing density along lines of sight")
        
        for i in tqdm(range(len(ells)), desc='Lines of sight'):
            for j in tqdm(range(len(distance)), desc=f'Distance number {i}', leave=False):
                data = torch.Tensor([cosell[i], sinell[i], 2.*distance[j]/self.max_distance-1.]).float()
                los_dens_network[i, j] = self.network.forward(data.to(self.device))
                x = distance[j]*cosell[i]
                y = distance[j]*sinell[i]
                los_dens_true[i, j] = ModelHelper.compute_extinction_model_density(self.model, x, y, 0.)
                
        np.savez(self.dens_los_filename, ells=ells, distance=distance, los_dens_true=los_dens_true,
                 los_dens_network=los_dens_network
                 )
