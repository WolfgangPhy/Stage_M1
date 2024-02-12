import pickle
import os.path
import ExtinctionModel

class ExtinctionModelLoader:
    def __init__(self, fiducial_model="2DModel.pickle"):
        self.fiducial_model = fiducial_model
        self.newmodel = None
        self.model = None

    def check_existing_model(self):
        if os.path.isfile(self.fiducial_model):
            print("Using existing model")
            self.newmodel = False
        else:
            self.newmodel = True

    def create_new_model(self):
        self.model = ExtinctionModel(100)
        with open(self.fiducial_model, "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self):
        with open(self.fiducial_model, "rb") as file:
            self.model = pickle.load(file)

