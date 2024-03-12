import os
import json
import shutil


class FileHelper:
    """
    A utility class for file-related operations.

    # Methods:
        - `init_test_directory()`: Initializes a test directory based on parameters from 'Parameters.json' (Static)
        - `give_config_value(config_file_path, key)`: Retrieves a specific value from a given
            configuration file. (Static)
    """
    
    @staticmethod
    def init_test_directory():
        """
        Initializes a test directory based on parameters from 'Parameters.json'.

        Returns:
            `str`: The path to the config file in the test directory.
        """
        with open('Parameters.json') as param_file:
            parameters = json.load(param_file)
        
        directory_name = "_".join([f"{key}_{value}" for key, value in parameters.items() if key not
                                   in ["ext_loss_function_custom", "dens_loss_function_custom", "is_new_network"]]
                                  )
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        elif input("Directory already exists. Do you want to do calculations in this directory? (y/n): ") == "y":
            return os.path.join(directory_name, "Config.json")
        else:
            raise ValueError("Directory already exists")
            
        npz_directory = os.path.join(directory_name, "NpzFiles")
        torch_directory = os.path.join(directory_name, "PyTorchFiles")
        output_directory = os.path.join(directory_name, "OutputFiles")
        plot_directory = os.path.join(directory_name, "Plots")
        
        os.makedirs(npz_directory)
        os.makedirs(torch_directory)
        os.makedirs(output_directory)
        os.makedirs(plot_directory)
        
        shutil.copy('Parameters.json', directory_name)
        shutil.copy('Config.json', directory_name)
        
        with open(os.path.join(directory_name, 'Config.json')) as param_file:
            config_data = json.load(param_file)
            config_data['datafile'] = os.path.join('./', directory_name, config_data['datafile'][2:])
            config_data['outfile'] = os.path.join('./', directory_name, config_data['outfile'][2:])
            config_data['logfile'] = os.path.join('./', directory_name, config_data['logfile'][2:])
            config_data['lossfile'] = os.path.join('./', directory_name, config_data['lossfile'][2:])
            config_data['valfile'] = os.path.join('./', directory_name, config_data['valfile'][2:])
            config_data['ext_grid_file'] = os.path.join('./', directory_name, config_data['ext_grid_file'][2:])
            config_data['dens_grid_file'] = os.path.join('./', directory_name, config_data['dens_grid_file'][2:])
            config_data['ext_los_file'] = os.path.join('./', directory_name, config_data['ext_los_file'][2:])
            config_data['dens_los_file'] = os.path.join('./', directory_name, config_data['dens_los_file'][2:])
            config_data['density_plot'] = os.path.join('./', directory_name, config_data['density_plot'][2:])
            config_data['extinction_plot'] = os.path.join('./', directory_name, config_data['extinction_plot'][2:])
            config_data['extinction_los_plot'] = os.path.join('./', directory_name,
                                                              config_data['extinction_los_plot'][2:]
                                                              )
            config_data['loss_plot'] = os.path.join('./', directory_name, config_data['loss_plot'][2:])
            
            with open(os.path.join(directory_name, 'Config.json'), 'w') as new_config_file:
                json.dump(config_data, new_config_file, indent=4)
           
        return os.path.join(directory_name, "Config.json")

    @staticmethod
    def give_config_value(config_file_path, key):
        """
        Retrieves a specific value from a given configuration file.

        Args:
        - `config_file_path (str)`: Path to the configuration file.
        - `key (str)`: Key for the desired value in the configuration file.

        Returns:
        - `Any`: The value associated with the specified key in the configuration file.
        """
        with open(config_file_path) as config_file:
            config = json.load(config_file)
        
        value = config[key]
        
        config_file.close()
        return value
