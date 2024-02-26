import MainTrainer as MainTrainer
import CreateDataFile as Creator
import ModelCalculator as Calculator
import ExtinctionModelLoader as Loader
import torch
import time
import json
import torch.nn.functional as F
import numpy as np
import CustomLossFunctions as LossFunctions
import ExtinctionNeuralNetBuilder as Builder
import FileHelper as FHelper
import ModelVisualizer as Visualizer

class MainProgram:
    """
    A class representing the main program for training an extinction model.

    # Attributes:
        - `nu_ext (float)`: Coefficient for extinction loss.
        - `nu_dens (float)`: Coefficient for density loss.
        - `int_loss_function (function)`: Loss function for extinction.
        - `dens_loss_function (function)`: Loss function for density.
        - `star_number (int)`: Number of stars in the dataset.
        - `int_reduction_method (str)`: Reduction method for extinction loss.
        - `dens_reduction_method (str)`: Reduction method for density loss.
        - `epoch_number (int)`: Number of training epochs.
        - `learning_rate (float)`: Learning rate for optimization.
        - `config_file_path (str)`: Path to the current test configuration file.
        - `device (torch.device)`: Device for running the program.
        - `loader (ExtinctionModelLoader)`: Instance of the model loader.
        - `dataset (torch.Tensor)`: Dataset for training.
        - `maintrainer (MainTrainer)`: Instance of the main trainer.
        - `builder (ExtinctionNeuralNetBuilder)`: Instance of the neural network builder.
        - `network (torch.nn.Module)`: Neural network model.
        - `opti (torch.optim)`: Optimization method.
        - `hidden_size (int)`: Size of the hidden layer.
        - `max_distance (float)`: Maximum distance in the dataset.
        - `parameters (dict)`: Dictionary of current test parameters.
        
    # Methods:
        - `get_parameters_from_json()`: Loads parameters from the "Parameters.json" file.
        - `set_model()`: Sets the model for training.
        - `set_hidden_size()`: Sets the size of the hidden layer.
        - `get_max_distance()`: Sets the maximum distance in the dataset.
        - `get_parameters_from_json()`: Loads parameters from the "Parameters.json" file.
        - `load_dataset()`: Loads the dataset for training.	
        - `set_parameters()`: Sets the program's parameters using loaded values.
        - `check_and_assign_loss_function(loss_function, custom_loss_function)`: Assigns the loss function based on parameters.
        - `create_data_file()`: Creates a data file for training.
        - `train()`: Initiates and runs the training process.
        - `calculate_density_extinction()`: Calculates the density and extinction values.
        - `Visualize()`: Visualizes the results.
        - `execute()`: Executes the complete program, including loading parameters, setting them, creating a data file, and training.

    # Example:
        >>> if __name__ == "__main__":
        >>>    start_time = time.time()
            
        >>>    mainprogram = MainProgram()
        >>>    mainprogram.execute()
            
        >>>    process_time = time.time() - start_time
        >>>    print("Process time: ", round(process_time, 0), " seconds")
    """
    def __init__(self):
        self.get_parameters_from_json()
        self.set_parameters()
        self.config_file_path = FHelper.FileHelper.init_test_directory()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.set_model()
        
        
    def set_model(self):
        """
        Sets the model for training using the ExtinctionModelLoader class.
        
        if the model is new, it creates a new model, otherwise it loads the existing model.
        """
        model_filename = FHelper.FileHelper.give_config_value(self.config_file_path, "model_file")
        self.loader = Loader.ExtinctionModelLoader(model_filename)
        self.loader.check_existing_model()
        if self.loader.newmodel:
            self.loader.create_new_model()
        else:
            self.loader.load_model()
            
    def set_hidden_size(self):
        """
        Compute and sets the size of the hidden layer based on the dataset size.
        """
        dim = 2
        k = (np.log10(self.dataset.__len__()))**-4.33 * 16. * self.dataset.__len__()/(dim+2)*2.
        self.hidden_size = int(k)

    def get_max_distance(self):
        """
        Gets the maximum distance in the dataset.
        """
        self.max_distance = np.max(self.dataset.distance.cpu().numpy())
    
    def get_parameters_from_json(self):
        """
        Loads parameters from the "Parameters.json" file and assigns them to the MainProgram "parameters" attribute.
        """
        with open("Parameters.json") as f:
            self.parameters = json.load(f)
            
    def load_dataset(self):
        """
        Loads the dataset for training.
        """
        datafile_path = FHelper.FileHelper.give_config_value(self.config_file_path, "datafile")
        
        self.dataset = torch.load(datafile_path, map_location=self.device)
          
    def set_parameters(self):
        """
        Sets the program's parameters using loaded values in the "parameters" dictionary.
        """
        self.nu_ext = self.parameters["nu_ext"]
        self.nu_dens = self.parameters["nu_dens"]
        self.int_loss_function = self.check_and_assign_loss_function(self.parameters["int_loss_function"], self.parameters["int_loss_function_custom"])
        self.dens_loss_function = self.check_and_assign_loss_function(self.parameters["dens_loss_function"], self.parameters["dens_loss_function_custom"])
        self.star_number = self.parameters["star_number"]
        self.int_reduction_method = self.parameters["int_reduction_method"]
        self.dens_reduction_method = self.parameters["dens_reduction_method"]
        self.epoch_number = self.parameters["epoch_number"]
        self.learning_rate = self.parameters["learning_rate"]
        
    def check_and_assign_loss_function(self, loss_function, custom_loss_function):
        """
        Assigns the loss function based on parameters.
        
        This method returns a callable loss function based on the given parameters.
        If the loss_function is custome in return the corresponding callable from the CustomLossFunctions module.
        If the loss_function is not custom, it returns the corresponding callable from the torch.nn.functional module.

        Args:
            - `loss_function (str)`: The name of the loss function.
            - `custom_loss_function (bool)`: A flag indicating whether a custom loss function is used.

        Raises:
            - `ValueError`: If the loss function is unknown.

        Returns:
            `callable`: The assigned loss function.
        """
        if custom_loss_function:
            custom_loss_method = getattr(LossFunctions.CustomLossFunctions, loss_function, None)
            if custom_loss_method is not None and callable(custom_loss_method):
                return custom_loss_method
            else:
                raise ValueError(f"Méthode de perte personnalisée inconnue : {loss_function}")
        elif loss_function in dir(F) and callable(getattr(F, loss_function)):
            return getattr(F, loss_function)
        else:
            raise ValueError(f"Fonction de perte inconnue : {loss_function}")
            
    def create_data_file(self):
        """
        Creates a data file for training using the CreateDataFile class.
        """
        CreateDataFile = Creator.CreateDataFile(self.star_number, self.loader.model, self.config_file_path)
        CreateDataFile.execute()
        
    def train(self):
        """
        Initiates and runs the training process using MainTrainer class.
        """
        self.maintrainer = MainTrainer.MainTrainer(self.epoch_number, self.nu_ext, self.nu_dens, self.int_loss_function, 
                                                        self.dens_loss_function, self.int_reduction_method, self.dens_reduction_method,
                                                        self.learning_rate, self.device, self.dataset, self.builder, self.network, self.opti, self.hidden_size,
                                                        self.max_distance, self.config_file_path)
        self.maintrainer.run()
        
    def calculate_density_extinction(self):
        """
        Calculates the density and extinction values using the ModelCalculator class.
        """
        calculator = Calculator.ModelCalculator(self.loader.model, self.builder, 5.1, -5., 5.1, -5., 0.1, self.max_distance, self.device, self.network, self.config_file_path)
        calculator.density_extinction_grid()
        calculator.density_extinction_sight()
        
    def Visualize(self):
        """
        Visualize the results (saave the plot in the "Plots" subsirectory of the current test directory) using the ModelVisualizer class.
        """
        visualizer = Visualizer.ModelVisualizer(self.config_file_path,self.dataset, self.max_distance)
        visualizer.compare_densities()
        visualizer.compare_extinctions()
        visualizer.extinction_vs_distance()
        
    def execute(self):
        """
        Executes the complete program, including loading parameters, setting them, creating a data file, and training.
        """
        self.create_data_file()
        self.load_dataset()
        self.set_hidden_size()
        self.builder = Builder.ExtinctionNeuralNetBuilder(self.device, self.hidden_size, self.learning_rate)
        self.network, self.opti = self.builder.create_net_integ(self.hidden_size)
        self.get_max_distance()
        self.train()
        self.calculate_density_extinction()
        self.Visualize()
        
if __name__ == "__main__":
    start_time = time.time()
    
    mainprogram = MainProgram()
    mainprogram.execute()
    
    process_time = time.time() - start_time
    print("Process time: ", round(process_time, 0), " seconds")