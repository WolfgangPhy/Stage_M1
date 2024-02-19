import MainTrainer as MainTrainer
import CreateDataFile as Creator
import time
import json
import torch.nn.functional as F
import CustomLossFunctions as LossFunctions

class MainProgram:
    
    def __init__(self):
        pass
    
    def get_parameters_from_json(self):
        with open("Parameters.json") as f:
            self.parameters = json.load(f)
    
    def set_parameters(self):
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
        if custom_loss_function:
            # Assurez-vous que CustomLossFunctions est correctement importé
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
        CreateDataFile = Creator.CreateDataFile(self.star_number)
        CreateDataFile.execute()
        
    def train(self):
        maintrainer = MainTrainer.MainTrainer(self.epoch_number, self.nu_ext, self.nu_dens, self.int_loss_function, self.dens_loss_function, self.int_reduction_method, self.dens_reduction_method, self.learning_rate)
        maintrainer.run()
        
    def execute(self):
        self.get_parameters_from_json()
        self.set_parameters()
        self.create_data_file()
        self.train()
        
    def train_only(self):
        self.get_parameters_from_json()
        self.set_parameters()
        self.train()
        
if __name__ == "__main__":
    start_time = time.time()
    
    mainprogram = MainProgram()
    mainprogram.train_only()
    
    process_time = time.time() - start_time
    print("Process time: ", round(process_time, 0), " seconds")