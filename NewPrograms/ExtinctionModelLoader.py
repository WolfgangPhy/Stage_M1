import pickle
import os.path
import ExtinctionModel as ExtinctionModel

class ExtinctionModelLoader:
    """
    A utility class for loading and managing ExtinctionModel instances.

    # Args:
        - `fiducial_model (str, optional)`: File name for the fiducial model pickle file. Defaults to "2DModel.pickle".

    # Attributes:
        - `fiducial_model (str)`: File name for the fiducial model pickle file.
        - `newmodel (bool)`: Flag indicating whether a new model is being created.
        - `model (ExtinctionModel)`: The ExtinctionModel instance.

    # Methods:
        - `check_existing_model()`: Checks if the fiducial model file already exists.
        - `create_new_model()`: Creates a new ExtinctionModel instance and saves it to the fiducial model file.
        - `load_model()`: Loads the ExtinctionModel instance from the fiducial model file.
    """
    def __init__(self, fiducial_model_filename="2DModel.pickle"):
        self.fiducial_model_filename = fiducial_model_filename
        self.newmodel = None
        self.model = None

    def check_existing_model(self):
        """
        Checks if the fiducial model file already exists.

        If the file exists, sets the `newmodel` flag to False, indicating that the existing model will be used.
        If the file does not exist, sets the `newmodel` flag to True.
        """
        if os.path.isfile(self.fiducial_model_filename):
            print("Using existing model")
            self.newmodel = False
        else:
            self.newmodel = True

    def create_new_model(self):
        """
        Creates a new ExtinctionModel instance and saves it to the fiducial model file.
        """
        self.model = ExtinctionModel.ExtinctionModel(100)#TODO
        with open(self.fiducial_model_filename, "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self):
        """
        Loads the ExtinctionModel instance from the fiducial model file.
        """
        with open(self.fiducial_model_filename, "rb") as file:
            self.model = pickle.load(file)

