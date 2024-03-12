import pickle
import os.path
from ExtinctionModel import ExtinctionModel


class ModelLoader:
    """
    A utility class for loading and managing `ExtinctionModel` instances.

    # Args:
        - `fiducial_model_filename (str)`: File name for the dataset pickle file.

    # Attributes:
        - `fiducial_model_filename (str)`: File name for the dataset pickle file.
        - `is_new_model (bool)`: Flag indicating whether a new model is being created.
        - `model (ExtinctionModel)`: The `ExtinctionModel` instance.

    # Methods:
        - `check_existing_model()`: Checks if the dataset file already exists.
        - `create_new_model()`: Creates a new `ExtinctionModel` instance and saves it to the dataset file.
        - `load_model()`: Loads the `ExtinctionModel` instance from the dataset file.
    
    # Example:
        The following example demonstrates how to use the ModelLoader class to load and manage `ExtinctionModel`
        instances.
        >>> loader = ModelLoader("fiducial_model.pkl")
        >>> loader.check_existing_model()
        >>> if loader.is_new_model:
        >>>     loader.create_new_model()
        >>> else:
        >>>     loader.load_model()

    """
    def __init__(self, fiducial_model_filename):
        self.fiducial_model_filename = fiducial_model_filename
        self.is_new_model = None
        self.model = None

    def check_existing_model(self):
        """
        Checks if the dataset file already exists.

        # Remarks:
            If the file exists, sets the `is_new_model` flag to False, indicating that the existing model will be used.
            If the file does not exist, sets the `is_new_model` flag to True.
        """
        if os.path.isfile(self.fiducial_model_filename):
            print("Using existing model")
            self.is_new_model = False
        else:
            print("Creating new model")
            self.is_new_model = True

    def create_new_model(self):
        """
        Creates a new `ExtinctionModel` instance and saves it to the dataset file.
        """
        self.model = ExtinctionModel(15)
        with open(self.fiducial_model_filename, "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self):
        """
        Loads the `ExtinctionModel` instance from the dataset file.
        """
        with open(self.fiducial_model_filename, "rb") as file:
            self.model = pickle.load(file)
