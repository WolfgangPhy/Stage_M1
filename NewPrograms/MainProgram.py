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

    # Methods:
        - `get_parameters_from_json()`: Loads parameters from the "Parameters.json" file.
        - `set_parameters()`: Sets the program's parameters using loaded values.
        - `check_and_assign_loss_function(loss_function, custom_loss_function)`: Assigns the loss function based on parameters.
        - `create_data_file()`: Creates a data file for training.
        - `train()`: Initiates and runs the training process.
        - `execute()`: Executes the complete program, including loading parameters, setting them, creating a data file, and training.
        - `train_only()`: Executes the training process only, without creating a new data file.

    # Example:
        >>> if __name__ == "__main__":
        >>>    start_time = time.time()
            
        >>>    mainprogram = MainProgram()
        >>>    mainprogram.train_only()
            
        >>>    process_time = time.time() - start_time
        >>>    print("Process time: ", round(process_time, 0), " seconds")
    """
    def __init__(self):
        self.get_parameters_from_json()
        self.set_parameters()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.load_dataset()
        self.max_distance = np.max(self.dataset.dist.cpu().numpy())
        self.set_hidden_size()
        self.builder = Builder.ExtinctionNeuralNetBuilder(self.device, self.hidden_size, self.learning_rate)
        self.network, self.opti = self.builder.create_net_integ(self.hidden_size)
        self.set_model()
        
    def set_model(self):
        self.loader = Loader.ExtinctionModelLoader("2DModel.pickle")
        self.extinction_model_loader.check_existing_model()
        if self.extinction_model_loader.newmodel:
            self.extinction_model_loader.create_new_model()
        else:
            self.extinction_model_loader.load_model()
            
    def set_hidden_size(self):
        dim = 2
        k = (np.log10(self.dataset.__len__()))**-4.33 * 16. * self.dataset.__len__()/(dim+2)*2.
        self.hidden_size = int(k)
    
    def get_parameters_from_json(self):
        """
        Loads parameters from the "Parameters.json" file.
        """
        with open("Parameters.json") as f:
            self.parameters = json.load(f)
            
    def load_dataset(self):
        """
        Open the config file
        """
        with open('Config.json') as config_file:
            config = json.load(config_file)
        
        self.dataset = torch.load(config['datafile'], map_location=self.device)
        config_file.close()
        
    
    def set_parameters(self):
        """
        Sets the program's parameters using loaded values.
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

        Args:
            - `loss_function (str)`: The name of the loss function.
            - `custom_loss_function (bool)`: A flag indicating whether a custom loss function is used.

        Raises:
            - `ValueError`: If the loss function is unknown.

        Returns:
            - `function`: The assigned loss function.
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
        Creates a data file for training.
        """
        CreateDataFile = Creator.CreateDataFile(self.star_number, self.loader.model)
        CreateDataFile.execute()
        
    def train(self):
        """
        Initiates and runs the training process.
        """
        self.maintrainer = MainTrainer.MainTrainer(self.epoch_number, self.nu_ext, self.nu_dens, self.int_loss_function, 
                                                        self.dens_loss_function, self.int_reduction_method, self.dens_reduction_method,
                                                        self.learning_rate, self.device, self.dataset, self.builder, self.network, self.opti, self.hidden_size,
                                                        self.max_distance)
        self.maintrainer.run()
        
    def calculate_density_extinction(self):
        calculator = Calculator.ModelCalculator(self.loader.model, self.builder, 5.1, -5., 5.1, -5., 0.1, self.max_distance, self.device, self.network)
        calculator.density_extinction_grid()
        calculator.density_extinction_sight()
        
    def execute(self):
        """
        Executes the complete program, including loading parameters, setting them, creating a data file, and training.
        """
        self.create_data_file()
        self.train()
        
if __name__ == "__main__":
    start_time = time.time()
    
    mainprogram = MainProgram()
    #mainprogram.train()
    mainprogram.calculate_density_extinction()
    
    process_time = time.time() - start_time
    print("Process time: ", round(process_time, 0), " seconds")