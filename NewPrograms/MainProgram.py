import MainTrainer as MainTrainer
import CreateDataFile as Creator
import time
import json
import torch.nn.functional as F
import CustomLossFunctions as LossFunctions

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
    
    def get_parameters_from_json(self):
        """
        Loads parameters from the "Parameters.json" file.
        """
        with open("Parameters.json") as f:
            self.parameters = json.load(f)
    
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
        CreateDataFile = Creator.CreateDataFile(self.star_number)
        CreateDataFile.execute()
        
    def train(self):
        """
        Initiates and runs the training process.
        """
        maintrainer = MainTrainer.MainTrainer(self.epoch_number, self.nu_ext, self.nu_dens, self.int_loss_function, self.dens_loss_function, self.int_reduction_method, self.dens_reduction_method, self.learning_rate)
        maintrainer.run()
        
    def execute(self):
        """
        Executes the complete program, including loading parameters, setting them, creating a data file, and training.
        """
        self.create_data_file()
        self.train()
        
if __name__ == "__main__":
    start_time = time.time()
    
    mainprogram = MainProgram()
    mainprogram.train()
    
    process_time = time.time() - start_time
    print("Process time: ", round(process_time, 0), " seconds")