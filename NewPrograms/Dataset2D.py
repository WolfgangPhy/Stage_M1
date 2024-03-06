import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset2D(Dataset):
    """
    PyTorch dataset for 2D extinction data.

    # Args:
        `ell (numpy.ndarray)`: Array of Galactic longitudes in degrees.
        `distance (numpy.ndarray)`: Array of distances in kiloparsecs (kpc).
        `K (numpy.ndarray)`: Array of total Absorption values.
        `error (numpy.ndarray)`: Array of errors on Absorption values.

    # Attributes:
        `ell (numpy.ndarray)`: Array of Galactic longitudes in degrees.
        `cosell (numpy.ndarray)`: Cosine of Galactic longitudes.
        `sinell (numpy.ndarray)`: Sine of Galactic longitudes.
        `distance (numpy.ndarray)`: Array of distances in kiloparsecs (kpc).
        `K (numpy.ndarray)`: Array of total Absorption values.
        `error (numpy.ndarray)`: Array of errors on Absorption values.

    # Methods:
        - `__len__()` : Returns the size of the dataset.
        - `__getitem__(index)`: Returns the sample at the given index.
        
    # Examples:
        Create an instance of the dataset:

        >>> dataset = Dataset2D(ell_data, distance_data, K_data, error_data)
        
        Access the length of the dataset:

        >>> len(dataset)
        
        Access a sample from the dataset:

        >>> sample = dataset[0]
    """
    def __init__(self, ell, distance, K, error):
        self.ell = ell
        self.cosell = np.cos(self.ell * np.pi/180.)
        self.sinell = np.sin(self.ell * np.pi/180.)
        self.distance = distance
        self.K = K
        self.error = error
        
    def __len__(self):
        """
        Returns the size of the dataset
        
        Returns:
            `float`: size of the dataset
        """
        return len(self.ell)
    
    def __getitem__(self, index): 
        """
        Returns the sample at the given index

        Args:
            `index (int)`: Index of the sample to retrieve.

        Returns:
            `tuple[torch.tensor, torch.tensor]`: A tuple containing two torch tensors:
                - The first tensor contains the 2D coordinates of the sample 
                as (cos(ell), sin(ell), dist).
                - The second tensor contains the values associated with the sample 
                as (total_absorption, error_on_absorption).
        """
        return (torch.tensor((self.cosell[index], self.sinell[index], self.distance[index])),
                torch.tensor((self.K[index], self.error[index]))
                )
