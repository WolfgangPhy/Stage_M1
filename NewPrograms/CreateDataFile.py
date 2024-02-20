import torch
import multiprocessing as mp
import ParallelProcessor as pp

class CreateDataFile:
    """
    Class for creating a data file based on an extinction model.

    # Args:
        - `star_number (int)`: Number of stars to be used in the data file.
        - `model (ExtinctionModel)`: The extinction model to be used for creating the data file.

    # Attributes:
        - `star_number (int)`: Number of stars to be used in the data file.
        - `model (ExtinctionModel)`: The extinction model to be used for creating the data file.

    # Methods:
        - `execute()`: Executes the process of creating the data file.

    # Examples:
        # Create an instance of CreateDataFile
        >>> data_creator = CreateDataFile(star_number, model)

        # Execute the data file creation process
        >>> data_creator.execute()
    """
    def __init__(self, star_number, model):
        self.star_number = star_number
        self.model = model

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
        dataset = pp.ParallelProcessor.process_parallel(
            self.model, pool, self.star_number, device, dtype
        )

        # Close the pool
        pool.close()

        # Save the dataset
        torch.save(dataset, "./PyTorchFiles/fiducial_model2D.pt") #TODO

        print("Done")
