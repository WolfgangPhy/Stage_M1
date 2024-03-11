import sys
import time
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from NetworkTrainer import NetworkTrainer
from NetworkHelper import NetworkHelper
from FileHelper import FileHelper


class MainTrainer:
    """
    Main program for training a neural network for extinction and density estimation.

    This class encapsulates the main workflow for training a neural network, including configuring,
    setting up log files, preparing the dataset, creating the network, initializing training, and executing
    training steps.
    
    # Args:
        - `epoch_number (int)`: Number of epochs for training the neural network.
        - `nu_ext (float)`: Lagrange multiplier for extinction loss calculation.
        - `nu_dens (float)`: Lagrange multiplier for density loss calculation.
        - `ext_loss_function (callable)`: Loss function for extinction estimation.
        - `dens_loss_function (callable)`: Loss function for density estimation.
        - `ext_reduction_method (str)`: Method for reducing the extinction loss.
        - `dens_reduction_method (str)`: Method for reducing the density loss.
        - `learning_rate (float)`: Learning rate for updating network parameters.
        - `device (torch.device)`: Computing device for running the neural network.
        - `dataset (torch.Dataset)`: Dataset containing training and validation samples.
        - `network (ExtinctionNetwork)`: Neural network model for extinction and density estimation.
        - `opti (torch.optim.Adam)`: Adam optimizer for updating network parameters.
        - `hidden_size (int)`: Size of the hidden layer in the neural network.
        - `max_distance (float)`: Maximum distance in the dataset.
        - `config_file_path (str)`: Path to the configuration file.
        - `batch_size (int)`: Size of the minibatches for training and validation.

    # Attributes:
        - `logfile (file)`: Logfile for recording training progress and results.
        - `dataset (torch.Dataset)`: Dataset containing training and validation samples.
        - `trainer (NetworkTrainer)`: Trainer for the extinction neural network.
        - `train_loader (torch.utils.data.DataLoader)`: Dataloader for training minibatches.
        - `val_loader (torch.utils.data.DataLoader)`: Dataloader for validation minibatches.
        - `network (ExtinctionNetwork)`: Neural network model for extinction and density estimation.
        - `opti (torch.optim.Adam)`: Adam optimizer for updating network parameters.
        - `epoch_number (int)`: Number of epochs for training the neural network.
        - `epoch (int)`: Current epoch in the training process.
        - `nu_ext (float)`: Lagrange multiplier for extinction loss calculation.
        - `nu_dens (float)`: Lagrange multiplier for density loss calculation.
        - `device (torch.device)`: Computing device for running the neural network.
        - `loss_ext_total (float)`: Total loss for extinction estimation.
        - `loss_dens_total (float)`: Total loss for density estimation.
        - `ext_loss_function (callable)`: Loss function for extinction estimation.
        - `dens_loss_function (callable)`: Loss function for density estimation.
        - `ext_reduction_method (str)`: Method for reducing the extinction loss.
        - `dens_reduction_method (str)`: Method for reducing the density loss.
        - `learning_rate (float)`: Learning rate for updating network parameters.
        - `hidden_size (int)`: Size of the hidden layer in the neural network.
        - `max_distance (float)`: Maximum distance in the dataset.
        - `config_file_path (str)`: Path to the configuration file.
        - `datafile_path (str)`: Path to the dataset file.
        - `outfile_path (str)`: Path to the output file for storing trained models.
        - `logfile_path (str)`: Path to the logfile for recording training progress and results.
        - `lossfile_path (str)`: Path to the logfile for recording training loss values.
        - `valfile_path (str)`: Path to the logfile for recording validation loss values.
        - `batch_size (int)`: Size of the minibatches for training and validation.
        
    # Methods:
        - `init_files_path()`: Initializes the path for the files used in the training process. 
        - `setup_logfile()`: Sets up the logfile for recording training progress and results.
        - `setup_csv_files()`: Sets up CSV files for recording training and validation loss values.
        - `prepare_dataset()`: Loads and preprocesses the dataset for training and validation.
        - `create_network()`: Creates the neural network architecture based on the configuration.
        - `init_training()`: Initializes the training process, setting up epoch-related variables.
        - `train_network()`: Performs the training iterations, updating the neural network parameters.
        - `run()`: Executes the main program, orchestrating the entire training process.
    """
    
    def __init__(self, epoch_number, nu_ext, nu_dens, ext_loss_function, dens_loss_function, ext_reduction_method,
                 dens_reduction_method, learning_rate, device, dataset, network, opti, hidden_size,
                 max_distance, config_file_path, batch_size):
        self.val_dens_total = None
        self.val_ext_total = None
        self.loss_dens_total = None
        self.loss_ext_total = None
        self.trainer = None
        self.val_loader = None
        self.train_loader = None
        self.logfile = None
        self.valfile_path = None
        self.lossfile_path = None
        self.logfile_path = None
        self.outfile_path = None
        self.datafile_path = None
        self.epoch = -1
        self.epoch_number = epoch_number
        self.nu_ext = nu_ext
        self.nu_dens = nu_dens
        self.ext_loss_function = ext_loss_function
        self.dens_loss_function = dens_loss_function
        self.ext_reduction_method = ext_reduction_method
        self.dens_reduction_method = dens_reduction_method
        self.learning_rate = learning_rate
        self.device = device
        self.dataset = dataset
        self.network = network
        self.opti = opti
        self.hidden_size = hidden_size
        self.max_distance = max_distance
        self.config_file_path = config_file_path
        self.batch_size = batch_size
        
    def init_files_path(self):
        """
        Initialize the path for the files using the current test configuration file.
        """
        self.datafile_path = FileHelper.give_config_value(self.config_file_path, 'datafile')
        self.outfile_path = FileHelper.give_config_value(self.config_file_path, 'outfile')
        self.logfile_path = FileHelper.give_config_value(self.config_file_path, 'logfile')
        self.lossfile_path = FileHelper.give_config_value(self.config_file_path, 'lossfile')
        self.valfile_path = FileHelper.give_config_value(self.config_file_path, 'valfile')
                
    def setup_logfile(self):
        """
        Set up the logfile for logging information during training.

        This method initializes and configures the log file based on the configuration settings.
        It writes information about Python and PyTorch versions, CUDA devices (if available),
        the selected computing device, the specified datafile, and the expected storage location for maps.

        """
        self.logfile = open(self.logfile_path, 'w')

        self.logfile.write('__Python VERSION:'+sys.version)
        self.logfile.write('__pyTorch VERSION:'+torch.__version__+'\n')
        if torch.cuda.is_available():
            self.logfile.write('__Number CUDA Devices:'+str(torch.cuda.device_count())+'\n')
            self.logfile.write('Active CUDA Device: GPU'+str(torch.cuda.current_device())+'\n')
            self.logfile.write('Available devices '+str(torch.cuda.device_count())+'\n')
            self.logfile.write('Current cuda device '+str(torch.cuda.current_device())+'\n')  
        self.logfile.write('Using device: '+str(self.device)+'\n')
        self.logfile.write('Using datafile: '+self.datafile_path+'\n')
        self.logfile.write('Maps stored in '+self.outfile_path+'_e*.pt\n')
          
    def setup_csv_files(self):
        """
        Set up CSV files for logging training and validation metrics.

        This method creates and initializes CSV files for logging training and validation metrics. 
        It opens the specified files, writes the header row, and then closes the files.

        # Files created:
            - `lossfile`: CSV file for training metrics.
                - Header: ['Epoch', 'TotalLoss', 'ExtinctionLoss', 'DensityLoss', 'Time']

            - `valfile`: CSV file for validation metrics.
                - Header: ['Epoch', 'TotalValLoss', 'ExtinctionValLoss', 'DensityValLoss', 'ValTime', 'TotalValTime']
        """
        with open(self.lossfile_path, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Epoch', 'TotalLoss', 'ExtinctionLoss', 'DensityLoss', 'Time'])
            csvfile.close()
        
        with open(self.valfile_path, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Epoch', 'TotalValLoss', 'ExtinctionValLoss', 'DensityValLoss', 'ValTime',
                                 'TotalValTime']
                                )
            csvfile.close()
           
    def prepare_dataset(self):
        """
        Prepare the dataset for training and validation.

        This method loads the dataset from the specified datafile, computes normalization coefficients,
        and prepares training and validation datasets along with corresponding data loaders.

        """

        # Limits of the dataset of our sample
        min_distance = 0.

        # Computes normalisation coefficients to help training, xmin and ymin both =0
        for i in range(self.dataset.__len__()):
            if self.dataset.distance[i].item() > self.max_distance:
                self.max_distance = self.dataset.distance[i].item()
                
        self.logfile.write('Distance normalisation factor (max_distance):'+str(self.max_distance)+'\n')  

        for i in range(self.dataset.__len__()):
            self.dataset.distance[i] = 2.*self.dataset.distance[i].item()/(self.max_distance-min_distance)-1.

        # prepares the dataset used for training and validation
        size = self.dataset.__len__()
        training_size = int(size*0.75)
        validation_size = size-training_size
        train_dataset, val_dataset = random_split(self.dataset, [training_size, validation_size])

        # prepares the dataloader for minibatch
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

    def create_network(self):
        """
        Create the neural network.

        This method initializes and configures the neural network for extinction and density estimation.
        The network is created with a hidden layer size determined by a formula based on the dataset size.

        """
        self.network.apply(NetworkHelper.init_weights)
        self.network.to(self.device) 
        self.network.train() # Set the network to training mode
        
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
        self.trainer = NetworkTrainer(self.network, self.device, self.opti,self.ext_loss_function,
                                                  self.dens_loss_function, self.ext_reduction_method,
                                                  self.dens_reduction_method
                                                  )
        for idx in tqdm(range(self.epoch_number+1)):
        
            # set start time of epoch and epoch number
            t0 = time.time()
            self.epoch = self.epoch+1

            # initialize variables at each epoch to store full losses
            self.loss_ext_total = 0.
            self.loss_dens_total = 0.

            # loop over minibatches
            nbatch = 0
            for in_batch, tar_batch in self.train_loader:
                nbatch += 1
                self.loss_ext_total, self.loss_dens_total = self.trainer.take_step(in_batch, tar_batch,
                                                                                   self.loss_ext_total,
                                                                                   self.loss_dens_total,
                                                                                   self.nu_ext, self.nu_dens
                                                                                   )
            
            # add up loss function contributions
            full_loss = self.loss_dens_total+self.loss_ext_total  # /(nbatch*1.) TODO
            # loss of the integral is on the mean, so we need to divide by the number of batches

            t1 = time.time()

            with open(self.lossfile_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([self.epoch, full_loss, self.loss_ext_total, self.loss_dens_total, (t1 - t0) / 60.])
            
            # compute loss on validation sample
            if self.epoch % 50 == 0:
                with torch.no_grad():  # turns off gradient calculation
                    # init loss variables
                    self.val_ext_total = 0.
                    self.val_dens_total = 0.

                    # loop over minibatches of validation sample
                    nbatch = 0
                    for in_batch_validation_set, tar_batch_validation_set in self.val_loader:
                        nbatch = nbatch+1
                        self.val_ext_total, self.val_dens_total = self.trainer.validation(in_batch_validation_set,
                                                                                          tar_batch_validation_set,
                                                                                          self.nu_ext, self.nu_dens,
                                                                                          self.val_ext_total,
                                                                                          self.val_dens_total
                                                                                          )
                
                    val_loss = self.val_dens_total+self.val_ext_total/(nbatch*1.)
                
                    t2 = time.time()

                    with open(self.valfile_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([self.epoch, val_loss, self.val_ext_total, self.val_dens_total,
                                             (t2 - t1) / 60., (t2 - t0) / 60.]
                                            )
            
            # save model every 10000 epoch and last step
            if self.epoch % 10000 == 0 or self.epoch == self.epoch_number:
                fname1 = '{}_e{}.pt'.format(self.outfile_path, self.epoch)
                self.network.to('cpu')
                torch.save({
                    'epoch': self.epoch,
                    'integ_state_dict': self.network.state_dict(),
                    'opti_state_dict': self.opti.state_dict()
                }, fname1)
                self.network.to(self.device)

        t1 = time.time()
        self.logfile = open(self.logfile_path, 'a')
        self.logfile.write("Total time (hours): "+str((t1-tstart)/3600.)+"\n")
        self.logfile.close()
        
    def run(self):
        """
        Execute the main trainer
        """

        self.init_files_path()
        self.setup_logfile()
        self.prepare_dataset()
        self.setup_csv_files()
        self.create_network()
        self.init_training()
        self.train_network()
        