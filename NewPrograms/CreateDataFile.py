import torch
import multiprocessing as mp
import ExtinctionModelLoader as Loader
import ModelVisualizer as Visualizer
import ParallelProcessor as pp

class CreateDataFile:
    """
    Class for creating a data file based on an extinction model.

    # Args:
        `fiducial_model (str, optionnal)`: Path to the fiducial model file (default: "2DModel.pickle").

    # Attributes:
        `extinction_model_loader (ExtinctionModelLoader)`: Loader for the extinction model.
        `model_visualizer (ModelVisualizer)`: Visualizer for the extinction model.

    # Methods:
        - `execute()`: Executes the process of creating the data file.

    # Examples:
        # Create an instance of CreateDataFile
        >>> data_creator = CreateDataFile(fiducial_model="custom_model.pickle")

        # Execute the data file creation process
        >>> data_creator.execute()
    """
    def __init__(self, fiducial_model="2DModel.pickle"):
        self.extinction_model_loader = Loader.ExtinctionModelLoader(fiducial_model)
        self.model_visualizer = Visualizer.ModelVisualizer()

    def execute(self):
        """
        Executes the process of creating the data file.
        """
        self.extinction_model_loader.check_existing_model()

        # Create new model if it doesn't exist
        if self.extinction_model_loader.newmodel:
            self.extinction_model_loader.create_new_model()
        else:
            self.extinction_model_loader.load_model()

        # Visualize the 2D model
        self.model_visualizer.visualize_model(self.extinction_model_loader.model)

        dtype = torch.float
        device = torch.device("cpu")
        processor_num = mp.cpu_count()

        # Set up multiprocessing pool
        pool = mp.Pool(processor_num)

        n = 10000 # Number of data points
        
        # Process the model in parallel and get the dataset
        dataset = pp.ParallelProcessor.process_parallel(
            self.extinction_model_loader.model, pool, n, device, dtype
        )

        # Close the pool
        pool.close()

        # Save the dataset
        torch.save(dataset, "./PyTorchFiles/fiducial_model2D.pt")

        print("Done")
