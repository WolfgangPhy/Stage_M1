import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import time
import ExtinctionNeuralNet as NeuralNet
import ExtinctionNeuralNetBuilder as Builder
import ExtinctionNeuralNetTrainer as Trainer
import json
import csv
from tqdm import tqdm


class MainTrainer:
    """
    Main program for training a neural network for extinction and density estimation.

    This class encapsulates the main workflow for training a neural network, including configuring,
    setting up log files, preparing the dataset, creating the network, initializing training, and executing training steps.
    
    # Args:
        - `epoch_number (int)`: Number of epochs for training the neural network.
        - `nu_ext (float)`: Lagrange multiplier for extinction loss calculation.
        - `nu_dens (float)`: Lagrange multiplier for density loss calculation.
        - `int_loss_function (callable)`: Loss function for extinction estimation.
        - `dens_loss_function (callable)`: Loss function for density estimation.
        - `int_reduction_method (str)`: Method for reducing the extinction loss.
        - `dens_reduction_method (str)`: Method for reducing the density loss.
        - `device (torch.device)`: Computing device for running the neural network.
        - `learning_rate (float)`: Learning rate for updating network parameters.
        - `dataset (torch.Dataset)`: Dataset containing training and validation samples.
        - `builder (ExtinctionNeuralNetBuilder)`: Builder for creating the neural network.
        - `network (ExtinctionNeuralNet)`: Neural network model for extinction and density estimation.
        - `opti (torch.optim.Adam)`: Adam optimizer for updating network parameters.
        - `hidden_size (int)`: Size of the hidden layer in the neural network.
        - `max_distance (float)`: Maximum distance in the dataset.

    # Attributes:
        - `config (dict)`: Configuration parameters for the training process.
        - `config_file (file)`: Configuration file for reading training parameters.
        - `logfile (file)`: Logfile for recording training progress and results.
        - `lossfile (file)`: Logfile for recording training loss values.
        - `valfile (file)`: Logfile for recording validation loss values.
        - `dataset (torch.Dataset)`: Dataset containing training and validation samples.
        - `train_loader (torch.utils.data.DataLoader)`: Dataloader for training minibatches.
        - `val_loader (torch.utils.data.DataLoader)`: Dataloader for validation minibatches.
        - `network (ExtinctionNeuralNet)`: Neural network model for extinction and density estimation.
        - `opti (torch.optim.Adam)`: Adam optimizer for updating network parameters.
        - `epoch (int)`: Current epoch in the training process.
        - `nu_ext (float)`: Lagrange multiplier for extinction loss calculation.
        - `nu_dens (float)`: Lagrange multiplier for density loss calculation.
        - `device (torch.device)`: Computing device for running the neural network.
        - `int_loss_function (callable)`: Loss function for extinction estimation.
        - `dens_loss_function (callable)`: Loss function for density estimation.
        - `int_reduction_method (str)`: Method for reducing the extinction loss.
        - `dens_reduction_method (str)`: Method for reducing the density loss.
        - `learning_rate (float)`: Learning rate for updating network parameters.
        - `builder (ExtinctionNeuralNetBuilder)`: Builder for creating the neural network.
        - `hidden_size (int)`: Size of the hidden layer in the neural network.
        - `max_distance (float)`: Maximum distance in the dataset.
        
    # Methods:
        - `open_config_file()`: Opens the configuration file and reads the training parameters.
        - `close_config_file()`: Closes the configuration file. - `OpenAppendConfigFile`: Opens the configuration file in append mode for additional logging.
        - `setup_logfile()`: Sets up the logfile for recording training progress and results.
        - `prepare_dataset()`: Loads and preprocesses the dataset for training and validation.
        - `create_network()`: Creates the neural network architecture based on the configuration.
        - `init_training()`: Initializes the training process, setting up epoch-related variables.
        - `train_network()`: Performs the training iterations, updating the neural network parameters.
        - `run()`: Executes the main program, orchestrating the entire training process.


    # Example:
        >>> # Example usage of MainProgram
        >>> main_program = MainProgram()
        >>> main_program.Execute()
        ```

    """
    
    def __init__(self, epoch_number, nu_ext, nu_dens, int_loss_function, dens_loss_function, int_reduction_method, 
                    dens_reduction_method, learning_rate, device, dataset, builder, network, opti, hidden_size, max_distance):
        self.epoch = -1
        self.epoch_number = epoch_number
        self.nu_ext = nu_ext
        self.nu_dens = nu_dens
        self.int_loss_function = int_loss_function
        self.dens_loss_function = dens_loss_function
        self.int_reduction_method = int_reduction_method
        self.dens_reduction_method = dens_reduction_method
        self.learning_rate = learning_rate
        self.device = device
        self.dataset = dataset
        self.builder = builder
        self.network = network
        self.opti = opti
        self.hidden_size = hidden_size
        self.max_distance = max_distance
        
    def open_config_file(self):
        """
        Open the config file
        """
        with open('Config.json') as self.config_file:
            self.config = json.load(self.config_file)
            
    def close_config_file(self):  
        """
        Close the config file
        """
        self.config_file.close()
            
    def setup_logfile(self):
        """
        Set up the logfile for logging information during training.

        This method initializes and configures the log file based on the configuration settings.
        It writes information about Python and PyTorch versions, CUDA devices (if available),
        the selected computing device, the specified datafile, and the expected storage location for maps.

        # Note:
            The log file is opened in 'write' mode if a new network is being used (newnet=True),
            and in 'append' mode otherwise.
            
        """
        if self.config['newnet']:
            self.logfile = open(self.config['logfile'],'w')
        else :
            self.logfile = open(self.config['logfile'],'a')
        
        dtype = torch.float

        self.logfile.write('__Python VERSION:'+sys.version)
        self.logfile.write('__pyTorch VERSION:'+torch.__version__+'\n')
        if torch.cuda.is_available():
            self.logfile.write('__Number CUDA Devices:'+str(torch.cuda.device_count())+'\n')
            self.logfile.write('Active CUDA Device: GPU'+str(torch.cuda.current_device())+'\n')
            self.logfile.write('Available devices '+str(torch.cuda.device_count())+'\n')
            self.logfile.write('Current cuda device '+str(torch.cuda.current_device())+'\n')  
        self.logfile.write('Using device: '+str(self.device)+'\n')
        self.logfile.write('Using datafile: '+self.config['datafile']+'\n')
        self.logfile.write('Maps stored in '+self.config['outfile']+'_e*.pt\n')
          
    def setup_csv_files(self):
        """
        Set up CSV files for logging training and validation metrics.

        This method creates and initializes CSV files for logging training and validation metrics. 
        It opens the specified files, writes the header row, and then closes the files.

        # Files created:
        - `lossfile`: CSV file for training metrics.
            - Header: ['Epoch', 'TotalLoss', 'IntegralLoss', 'DensityLoss', 'Time']

        - `valfile`: CSV file for validation metrics.
            - Header: ['Epoch', 'TotalValLoss', 'IntegralValLoss', 'DensityValLoss', 'ValTime', 'TotalValTime']
        """
        self.lossfile = open(self.config['lossfile'],'w')
        csv_writer = csv.writer(self.lossfile)
        csv_writer.writerow(['Epoch', 'TotalLoss', 'IntegralLoss', 'DensityLoss', 'Time'])
        self.lossfile.close()
        
        self.valfile = open(self.config['valfile'],'w')
        csv_writer = csv.writer(self.valfile)
        csv_writer.writerow(['Epoch', 'TotalValLoss', 'IntegralValLoss', 'DensityValLoss', 'ValTime', 'TotalValTime'])
        self.valfile.close()
           
    def prepare_dataset(self):
        """
        Prepare the dataset for training and validation.

        This method loads the dataset from the specified datafile, computes normalization coefficients,
        and prepares training and validation datasets along with corresponding data loaders.

        """

        # Limits of the dataset of our sample
        min_distance=0.

        # Computes normalisation coefficients to help training, xmin and ymin both =0
        for i in range(self.dataset.__len__()):
            if self.dataset.dist[i].item()> max_distance:
                max_distance = self.dataset.dist[i].item()
                
        self.logfile.write('Distance normalisation factor (max_distance):'+str(max_distance)+'\n')  

        for i in range(self.dataset.__len__()):
            self.dataset.dist[i] = 2.*self.dataset.dist[i].item()/(max_distance-min_distance)-1.

        # prepares the dataset used for training and validation
        size = self.dataset.__len__()
        training_size= int(size*0.75)
        validation_size = size-training_size
        train_dataset, val_dataset = random_split(self.dataset, [training_size, validation_size])

        # prepares the dataloader for minibatch
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=10000, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=10000,shuffle=False)
        
    def create_network(self):
        """
        Create the neural network.

        This method initializes and configures the neural network for extinction and density estimation.
        The network is created with a hidden layer size determined by a formula based on the dataset size.
        It also handles loading a pretrained network if specified in the configuration.

        """
        self.network.apply(self.builder.init_weights)
        self.network.to(self.device) 
        self.network.train()
        
    def init_training(self):
        """
        Initialize the training process.

        This method sets up the initial conditions for the training process.
        It initializes the training epoch and specifies Lagrange multipliers for loss calculations.
        The variables `nu_ext` and `nu_dens` are used by the loss function during training.
        The initialization details are logged into the logfile.

        """
        # initialize epoch
        try:
            self.epoch
        except NameError:
            self.epoch = -1

        self.logfile.write('(nu_ext,nu_dens)=('+str(self.nu_ext)+','+str(self.nu_dens)+')\n')
        self.logfile.close()
            
    def train_network(self):
        """
        Train the neural network.

        This method trains the neural network using the specified training configuration.
        It iterates through the specified number of epochs, performing training steps on each minibatch.
        The training progress, losses, and validation performance are logged into the logfile.

        """
        tstart = time.time()
        self.trainer = Trainer.ExtinctionNeuralNetTrainer(self.builder, self.int_loss_function, self.dens_loss_function, self.int_reduction_method, self.dens_reduction_method)
        for idx in tqdm(range(self.epoch_number+1)):
        
            # set start time of epoch and epoch number
            t0 = time.time()
            self.epoch = self.epoch+1

            # initialize variables at each epoch to store full losses
            self.lossint_total=0.
            self.lossdens_total=0.

            # loop over minibatches
            nbatch = 0
            for in_batch, tar_batch in self.train_loader:
                nbatch = nbatch+1
                self.lossint_total, self.lossdens_total = self.trainer.take_step(in_batch , tar_batch, self.lossint_total, self.lossdens_total, self.nu_ext, self.nu_dens)
            
            # add up loss function contributions
            full_loss = self.lossdens_total+self.lossint_total # /(nbatch*1.) # loss of the integral is on the mean so we need to divide by the number of batches

            # print progress for full batch
            #if epoch%10==0:
            t1 = time.time()
            
            with open(self.config['lossfile'], 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([self.epoch, full_loss, self.lossint_total, self.lossdens_total, (t1 - t0) / 60.])
            
            # compute loss on validation sample
            if self.epoch%50==0:
                with torch.no_grad(): # turns off gradient calculation
                    # init loss variables
                    self.valint_total=0.
                    self.valdens_total=0.

                    # loop over minibatches of validation sample
                    nbatch=0
                    for in_batch_validation_set,tar_batch_validation_set in self.val_loader:
                        nbatch=nbatch+1
                        self.valint_total, self.valdens_total = self.trainer.validation(in_batch_validation_set, tar_batch_validation_set, self.nu_ext, self.nu_dens, self.valint_total, self.valdens_total)
                
                    val_loss = self.valdens_total+self.valint_total/(nbatch*1.)
                
                    t2 = time.time()

                    with open(self.config['valfile'], 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([self.epoch, val_loss, self.valint_total, self.valdens_total, (t2 - t1) / 60., (t2 - t0) / 60.])
            
            # save model every 50 epoch and last step
            if self.epoch%10000==0 or self.epoch==self.epoch_number:
                fname1 = '{}_e{}.pt'.format(self.config['outfile'],self.epoch)
                self.network.to('cpu')
                torch.save({
                    'epoch': self.epoch,
                    'integ_state_dict': self.network.state_dict(),
                    'opti_state_dict': self.opti.state_dict()
                },fname1)
                self.network.to(self.device)

        t1=time.time()
        self.logfile=open(self.config['logfile'],'a')
        self.logfile.write("Total time (hours): "+str((t1-tstart)/3600.)+"\n")
        self.logfile.close()
        
    def run(self):
        """
        Execute the main trainer
        """
        
        self.open_config_file()
        self.setup_logfile()
        self.prepare_dataset()
        
        if self.config['newnet']:
            self.setup_csv_files()
            
        self.create_network()
        self.init_training()
        self.train_network()
        self.close_config_file()
        