import torch
import time
import json
import torch.nn.functional as f
import numpy as np
from MainTrainer import MainTrainer
from CreateDataFile import CreateDataFile
from Calculator import Calculator
from ModelLoader import ModelLoader
from CustomLossFunctions import CustomLossFunctions
from FileHelper import FileHelper
from Visualizer import Visualizer


class MainProgram:
    """
    A class representing the main program for training an extinction model.

    # Attributes:
        - `nu_ext (float)`: Coefficient for extinction loss.
        - `nu_dens (float)`: Coefficient for density loss.
        - `ext_loss_function (function)`: Loss function for extinction.
        - `dens_loss_function (function)`: Loss function for density.
        - `star_number (int)`: Number of stars in the dataset.
        - `ext_reduction_method (str)`: Reduction method for extinction loss.
        - `dens_reduction_method (str)`: Reduction method for density loss.
        - `epoch_number (int)`: Number of training epochs.
        - `learning_rate (float)`: Learning rate for optimization.
        - `config_file_path (str)`: Path to the current test configuration file.
        - `device (torch.device)`: Device for running the program.
        - `loader (ExtinctionModelLoader)`: Instance of the model loader.
        - `dataset (torch.Tensor)`: Dataset for training.
        - `maintrainer (MainTrainer)`: Instance of the main trainer.
        - `network (torch.nn.Module)`: Neural network model.
        - `opti (torch.optim)`: Optimization method.
        - `hidden_size (int)`: Size of the hidden layer.
        - `max_distance (float)`: Maximum distance in the dataset.
        - `parameters (dict)`: Dictionary of current test parameters.
        - `batch_size (int)`: Batch size for training.
        - `is_new_network (bool)`: Flag indicating whether the network is new.
        - `is_new_datafile (bool)`: Flag indicating whether the data file is new.
        - `checkpoint_epoch (int)`: Epoch number for checkpointing.

    # Methods:
        - `get_parameters_from_json()`: Loads parameters from the "Parameters.json" file.
        - `set_model()`: Sets the model for training.
        - `set_hidden_size()`: Sets the size of the hidden layer.
        - `get_max_distance()`: Sets the maximum distance in the dataset.
        - `load_dataset()`: Loads the dataset for training.
        - `set_parameters()`: Sets the program's parameters using loaded values.
        - `check_and_assign_loss_function(loss_function, custom_loss_function)`: Assigns the loss function based
            on parameters.
        - `create_data_file()`: Creates a data file for training.
        - `train()`: Initiates and runs the training process.
        - `calculate_density_extinction()`: Calculates the density and extinction values.
        - `Visualize()`: Visualizes the results.
        - `execute()`: Executes the complete program, including loading parameters, setting them, creating a data file,
            and training.

    # Example:
        The following example demonstrates how to use the MainProgram class to execute the complete program.

        >>> main_program = MainProgram()
        >>> main_program.execute()
    """

    def __init__(self):
        self.opti = None
        self.network = None
        self.main_trainer = None
        self.batch_size = None
        self.learning_rate = None
        self.epoch_number = None
        self.dens_reduction_method = None
        self.ext_reduction_method = None
        self.star_number = None
        self.dens_loss_function = None
        self.ext_loss_function = None
        self.nu_dens = None
        self.nu_ext = None
        self.dataset = None
        self.parameters = None
        self.max_distance = None
        self.hidden_size = None
        self.checkpoint_epoch = None
        self.loader = None
        self.is_new_network = None
        self.is_new_datafile = None
        self.get_parameters_from_json()
        self.set_parameters()
        self.config_file_path = FileHelper.init_test_directory()
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.set_model()

    def set_model(self):
        """
        Sets the model for training using the `ModelLoader` class.

        # Remarks:
            If the model is new, it creates a new model, otherwise it loads the existing model.
        """
        model_filename = FileHelper.give_config_value(
            self.config_file_path, "model_file"
        )
        self.loader = ModelLoader(model_filename)
        self.loader.check_existing_model()
        if self.loader.is_new_model:
            self.loader.create_new_model()
        else:
            self.loader.load_model()

    def set_hidden_size(self):
        """
        Compute and sets the size of the hidden layer based on the dataset size.
        """
        dim = 2
        k = (
            (np.log10(self.dataset.__len__())) ** -4.33
            * 16.0
            * self.dataset.__len__()
            / (dim + 2)
            * 2.0
        )
        self.hidden_size = int(k)

    def get_max_distance(self):
        """
        Gets the maximum distance in the dataset.
        """
        self.max_distance = np.max(self.dataset.distance.cpu().numpy())

    def get_parameters_from_json(self):
        """
        Loads parameters from the "Parameters.json" file and assigns them to the MainProgram `parameters` attribute.
        """
        with open("Parameters.json") as file:
            self.parameters = json.load(file)

    def load_dataset(self):
        """
        Loads the dataset for training.
        """
        datafile_path = FileHelper.give_config_value(self.config_file_path, "datafile")

        self.dataset = torch.load(datafile_path, map_location=self.device)

    def set_parameters(self):
        """
        Sets the program's parameters using loaded values in the `parameters` dictionary.
        """
        self.nu_ext = self.parameters["nu_ext"]
        self.nu_dens = self.parameters["nu_dens"]
        self.ext_loss_function = self.check_and_assign_loss_function(
            self.parameters["ext_loss_function"],
            self.parameters["ext_loss_function_custom"],
        )
        self.dens_loss_function = self.check_and_assign_loss_function(
            self.parameters["dens_loss_function"],
            self.parameters["dens_loss_function_custom"],
        )
        self.star_number = self.parameters["star_number"]
        self.ext_reduction_method = self.parameters["ext_reduction_method"]
        self.dens_reduction_method = self.parameters["dens_reduction_method"]
        self.epoch_number = self.parameters["epoch_number"]
        self.learning_rate = self.parameters["learning_rate"]
        self.batch_size = self.parameters["batch_size"]
        self.is_new_network = self.parameters["is_new_network"]
        self.checkpoint_epoch = self.parameters["checkpoint_epoch"]

    def check_and_assign_loss_function(self, loss_function, custom_loss_function):
        """
        Assigns the loss function based on parameters.

        # Remarks:
            This method returns a callable loss function based on the given parameters.
            If the loss_function is custom in return the corresponding callable from the `CustomLossFunctions` module.
            If the loss_function is not custom, it returns the corresponding callable from the `torch.nn.functional`
            module.

        # Important Note:
            Some of the loss functions does not take same parameters, for example, the loglike_loss function
            from the `CustomLossFunctions` module takes tar_batch parameters
            which contains the extinction and the sigma, while the `torch.nn.functional.mse_loss` function only takes
            the extinction as target.
            So be careful when using the loss functions, and make sure that the loss function you are using takes
            the right parameters.
            Some try/catch are implemented to avoid some errors and clarify the problem.

        # Args:
            - `loss_function (str)`: The name of the loss function.
            - `custom_loss_function (bool)`: A flag indicating whether a custom loss function is used.

        # Raises:
            - `ValueError`: If the loss function is unknown.

        # Returns:
            `callable`: The assigned loss function.
        """
        if custom_loss_function:
            custom_loss_method = getattr(CustomLossFunctions, loss_function, None)
            if custom_loss_method is not None and callable(custom_loss_method):
                return custom_loss_method
            else:
                raise ValueError(f"Unknown custom loss function : {loss_function}")
        elif loss_function in dir(f) and callable(getattr(f, loss_function)):
            return getattr(f, loss_function)
        else:
            raise ValueError(f"Unknown loss function : {loss_function}")

    def create_data_file(self):
        """
        Creates a data file for training using the `CreateDataFile` class.
        """
        creator = CreateDataFile(
            self.star_number, self.loader.model, self.config_file_path
        )
        creator.execute()

    def train(self):
        """
        Initiates and runs the training process using `MainTrainer` class.
        """
        self.main_trainer = MainTrainer(
            self.epoch_number,
            self.nu_ext,
            self.nu_dens,
            self.ext_loss_function,
            self.dens_loss_function,
            self.ext_reduction_method,
            self.dens_reduction_method,
            self.learning_rate,
            self.device,
            self.dataset,
            self.network,
            self.opti,
            self.hidden_size,
            self.max_distance,
            self.config_file_path,
            self.batch_size,
        )
        self.main_trainer.run()

    def calculate_density_extinction(self):
        """
        Calculates the density and extinction values using the `Calculator` class.
        """
        calculator = Calculator(
            self.loader.model,
            5.1,
            -5.0,
            5.1,
            -5.0,
            0.1,
            self.max_distance,
            self.device,
            self.network,
            self.config_file_path,
        )
        calculator.compute_extinction_grid()
        calculator.compute_extinction_sight()
        calculator.compute_density_grid()
        calculator.compute_density_sight()

    def visualize(self):
        """
        Visualize the results.

        # Remarks:
            This method visualizes the results using the `Visualizer` class and saves the plots in the "Plots"
            subdirectory of the current test directory.
        """
        visualizer = Visualizer(self.config_file_path, self.dataset, self.max_distance)
        visualizer.star_map()
        visualizer.plot_model()
        visualizer.compare_densities()
        visualizer.density_vs_distance()
        visualizer.compare_extinctions()
        visualizer.extinction_vs_distance()
        visualizer.loss_function()
        if self.is_new_datafile:
            visualizer.model_histogram()
        visualizer.network_density_histogram()
        visualizer.density_true_vs_network()
        visualizer.density_difference_vs_network()
        

    def execute(self):
        """
        Executes the complete program.

        # Remarks:
            This method executes the complete program, including loading parameters, setting them, creating a data file,
            and training.
        """
        if self.is_new_datafile:
            self.create_data_file()
        self.load_dataset()
        self.set_hidden_size()
        # self.network, self.opti = NetworkHelper.create_net_integ(self.hidden_size, self.device, self.learning_rate,
        #                                                         self.is_new_network, self.checkpoint_epoch,
        #                                                         self.config_file_path
        #                                                         )
        self.get_max_distance()
        if self.is_new_network:
            self.train()
        # self.calculate_density_extinction()
        self.visualize()  # TODO : Plots files sont écrasés à chaque exécution - Ajouter au README.md


if __name__ == "__main__":
    start_time = time.time()

    main_program = MainProgram()
    main_program.execute()

    process_time = time.time() - start_time
    print("Process time: ", round(process_time, 0), " seconds")
