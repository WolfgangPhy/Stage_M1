import numpy as np

class ExtinctionModel:
    """
    Class to store a given model.

    # Attributes:
        `m_tot (numpy.ndarray)`: Total mass for each model.
        `x0 (numpy.ndarray)`: Cartesian location of cloud along the x-axis [-4, 4] kpc for each model.
        `y0 (numpy.ndarray)`: Cartesian location of cloud along the y-axis [-4, 4] kpc for each model.
        `z0 (numpy.ndarray)`: Cartesian location of cloud along the z-axis [-0.5, 0.5] kpc for each model.
        `s1 (numpy.ndarray)`: Size along the first axis for each model in kpc.
        `s2 (numpy.ndarray)`: Size along the second axis for each model in kpc.
        `s3 (numpy.ndarray)`: Size along the third axis for each model in kpc.
        `a1 (numpy.ndarray)`: Orientation angle along the first axis [-30, 30] degrees for each model.
        `a2 (numpy.ndarray)`: Orientation angle along the second axis [-45, 45] degrees for each model.

    # Methods:
        - `__init__(N)`: Initializes an instance of the ExtinctionModel class.
        - `__len__` : Returns the total number of samples, which is the length of the 'm_tot' array.
    """
    def __init__(self, N):
        """
        Initializes an instance of the ExtinctionModel class.

        # Args:
            `N (int)`: Number of clouds to generate.
        """
        self.m_tot = np.random.rand(N)*0.1 
        self.x0 = np.random.rand(N)*8.-4. 
        self.y0 = np.random.rand(N)*8.-4. 
        self.z0 = np.random.rand(N)-0.5   
        self.s1 = 0.05+np.random.rand(N)*0.05 
        self.s2 = 0.05+np.random.rand(N)*0.05
        self.s3 = 0.05+np.random.rand(N)*0.05
        self.a1 = np.random.rand(N)*60.-30. 
        self.a2 = np.random.rand(N)*90.-45. 

    def __len__(self):
        """
        Returns the total number of samples.

        # Returns:
            `int`: The length of the 'm_tot' array.
        """
        return len(self.m_tot)
    
        

    