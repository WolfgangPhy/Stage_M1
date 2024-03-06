import torch
import multiprocessing as mp
import ParallelProcessor as ParaProcess
from FileHelper import FileHelper


class CreateDataFile:
    """
    Class for creating a data file based on an extinction model.

    # Args:
        - `star_number (int)`: Number of stars to be used in the data file.
        - `model (ExtinctionModel)`: The extinction model to be used for creating the data file.
        - `config_file_path (str)`: The path to the configuration file.

    # Attributes:
        - `star_number (int)`: Number of stars to be used in the data file.
        - `model (ExtinctionModel)`: The extinction model to be used for creating the data file.
        - `config_file_path (str)`: The path to the current test configuration file.

    # Methods:
        - `execute()`: Executes the process of creating the data file.
    """
    def __init__(self, star_number, model, config_file_path):
        self.star_number = star_number
        self.model = model
        self.config_file_path = config_file_path

    def execute(self):
        """
        Executes the process of creating the data file.
        """
        
        dtype = torch.float
        device = torch.device("cpu")
        processor_num = mp.cpu_count()

        # Set up multiprocessing pool
        pool = mp.Pool(processor_num)
        
        # Process the model in parallel and get the dataset
        dataset = ParaProcess.ParallelProcessor.process_parallel(
            self.model, pool, self.star_number, device, dtype
        )

        # Close the pool
        pool.close()

        dataset_filepath = FileHelper.give_config_value(self.config_file_path, "datafile")
        torch.save(dataset, dataset_filepath)

        print("Done")
