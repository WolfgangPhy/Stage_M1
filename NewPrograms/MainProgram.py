import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from scipy.spatial.transform import Rotation as R
import time
import ExtinctionNeuralNet
import ExtinctionNeuralNetBuilder
import json


class MainProgram:
    """
    Main program for training a neural network for extinction and density estimation.

    This class encapsulates the main workflow for training a neural network, including configuring,
    setting up log files, preparing the dataset, creating the network, initializing training, and executing training steps.

    # Attributes:
        - `config` (dict): Configuration parameters for the training process.
        - `logfile` (file): Logfile for recording training progress and results.
        - `dataset` (torch.Dataset): Dataset containing training and validation samples.
        - `train_loader` (torch.utils.data.DataLoader): Dataloader for training minibatches.
        - `val_loader` (torch.utils.data.DataLoader): Dataloader for validation minibatches.
        - `network` (ExtinctionNeuralNet): Neural network model for extinction and density estimation.
        - `opti` (torch.optim.Adam): Adam optimizer for updating network parameters.
        - `epoch` (int): Current epoch in the training process.
        
    # Methods:
        - `OpenConfigFile()`: Opens the configuration file and reads the training parameters.
        - `CloseConfigFile()`: Closes the configuration file. - `OpenAppendConfigFile`: Opens the configuration file in append mode for additional logging.
        - `SetupLogfile()`: Sets up the logfile for recording training progress and results.
        - `PrepareDataset()`: Loads and preprocesses the dataset for training and validation.
        - `CreateNetwork()`: Creates the neural network architecture based on the configuration.
        - `InitTraining()`: Initializes the training process, setting up epoch-related variables.
        - `TrainNetwork()`: Performs the training iterations, updating the neural network parameters.
        - `Execute()`: Executes the main program, orchestrating the entire training process.


    # Example:
        >>> # Example usage of MainProgram
        >>> main_program = MainProgram()
        >>> main_program.Execute()
        ```

    """
    def OpenConfigFile(self):
        """
        Open the config file
        """
        with open('Config.json') as json_file:
            self.config = json.load(json_file)
            
    def CloseConfigFile(self):  
        """
        Close the config file
        """
        self.config.close()
            
    def SetupLogfile(self):
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
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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
          
    def PrepareDataset(self):
        """
        Prepare the dataset for training and validation.

        This method loads the dataset from the specified datafile, computes normalization coefficients,
        and prepares training and validation datasets along with corresponding data loaders.

        """
        self.dataset = torch.load(self.config['datafile'], map_location='cpu')

        #attrs = vars(dataset)
        #print(', '.join("%s: %s" % item for item in attrs.items()))

        # Limits of the dataset of our sample
        xmin=0.
        xmax=0.

        # Computes normalisation coefficients to help training, xmin and ymin both =0
        for i in range(self.dataset.__len__()):
            if self.dataset.dist[i].item()> xmax:
                xmax = self.dataset.dist[i].item()
                
        self.logfile.write('Distance nomalisation factor (xmax):'+str(xmax)+'\n')  

        for i in range(self.dataset.__len__()):
            self.dataset.dist[i] = 2.*self.dataset.dist[i].item()/(xmax-xmin)-1.

        # prepares the dataset used for training and validation
        size = self.dataset.__len__()
        training_size= int(size*0.75)
        validation_size = size-training_size
        train_dataset, val_dataset = random_split(self.dataset, [training_size, validation_size])

        # prepares the dataloader for minibatch
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=10000, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=10000,shuffle=False)
        
    def CreateNetwork(self):
        """
        Create the neural network.

        This method initializes and configures the neural network for extinction and density estimation.
        The network is created with a hidden layer size determined by a formula based on the dataset size.
        It also handles loading a pretrained network if specified in the configuration.

        """
        dim = 2
        k = (np.log10(self.dataset.__len__()))**-4.33 * 16. * self.dataset.__len__()/(dim+2)*2.
        self.logfile.write('Using '+str(int(k))+' neurons in the hidden layer\n')
        hidden_size = int(k)

        self.network, self.opti = ExtinctionNeuralNetBuilder.create_net_integ(hidden_size,learning_rate=1e-3)
        self.network.apply(ExtinctionNeuralNetBuilder.init_weights)
        if not self.config['newnet']:
            # define networks
            self.network = ExtinctionNeuralNet(hidden_size)

            # load checkpoint file
            checkpoint = torch.load(self.config['pretrainednetwork'],map_location='cpu')

            # update networks to checkpoint state
            self.network.load_state_dict(checkpoint['integ_state_dict'])

            # update optimizers state
            self.opti.load_state_dict(checkpoint['opti_state_dict'])

            # update epoch
            self.epoch = checkpoint['epoch']

        self.network.to(self.device)    
        self.network.train()
        
    def InitTraining(self):
        """
        Initialize the training process.

        This method sets up the initial conditions for the training process.
        It initializes the training epoch and specifies Lagrange multipliers for loss calculations.
        The global variables `nu_ext` and `nu_dens` are used by the loss function during training.
        The initialization details are logged into the logfile.

        """
        # initialize epoch
        try:
            epoch
        except NameError:
            epoch = -1

        # Lagrange multipliers for loss calculation
        # global variables used by fullloss
        nu_ext = 1.   # total ext must match observations
        nu_dens = 1.   # density must be positive
        self.logfile.write('(nu_ext,nu_dens)=('+str(nu_ext)+','+str(nu_dens)+')\n\nStart training:\n')
        self.logfile.close()
            
    def TrainNetwork(self):
        """
        Train the neural network.

        This method trains the neural network using the specified training configuration.
        It iterates through the specified number of epochs, performing training steps on each minibatch.
        The training progress, losses, and validation performance are logged into the logfile.

        """
        tstart = time.time()
        for idx in range(self.epochs+1):
            #open logfile here and close it at end of epoch to make sure everything is written
            self.OpenAppendConfigFile()
        
            # set start time of epoch and epoch number
            t0 = time.time()
            epoch = epoch+1

            # initialize variables at each epoch to store full losses
            lossint_total=0.
            lossdens_total=0.

            # loop over minibatches
            nbatch = 0
            for xb,yb in self.train_loader:
                nbatch = nbatch+1
                ExtinctionNeuralNetBuilder.take_step(xb,yb)

            # add up loss function contributions
            full_loss = lossdens_total+lossint_total # /(nbatch*1.) # loss of the integral is on the mean so we need to divide by the number of batches

            # print progress for full batch
            #if epoch%10==0:
            t1 = time.time()
            self.logfile.write("Epoch "+str(epoch)+" -  Loss:"+str(full_loss)+" ("+str(lossint_total)+','+str(lossdens_total)+") Total time (min): "+str((t1-t0)/60.)+"\n")
    
            self.lossfile = open(self.config['lossfile'],'a')
            self.lossfile.write(str(epoch)+' '+str(full_loss)+' '+str(lossint_total)+' '+str(lossdens_total)+' '+str((t1-t0)/60.)+'\n')
            self.lossfile.close()
            
            # compute loss on validation sample
            if epoch%50==0:
                with torch.no_grad(): # turns off gradient calculation
                    # init loss variables
                    valint_total=0.
                    valdens_total=0.

                    # loop over minibatches of validation sample
                    nbatch=0
                    for x_val,y_val in self.val_loader:
                        nbatch=nbatch+1
                        ExtinctionNeuralNetBuilder.validation(x_val, y_val)
                
                    val_loss = valdens_total+valint_total/(nbatch*1.)
                
                    t2 = time.time()
                    self.logfile.write("Validation Epoch "+str(epoch)+" -  Loss:"+str(val_loss)+" ("+str(valint_total)+","+str(valdens_total)+") Time val (min):"+str((t2-t1)/60.)+" Total:"+str((t2-t0)/60.)+"\n")

                    valfile = open(self.config['valfile'],'a')
                    valfile.write(str(epoch)+' '+str(val_loss)+' '+str(valint_total)+' '+str(valdens_total)+' '+str((t2-t1)/60.)+' '+str((t2-t0)/60.)+'\n')
                    valfile.close()
                    

            # end of epoch close log file to make sure the information can be read about progress
            self.logfile.close()
            
            # save model every 50 epoch and last step
            if epoch%10000==0 or epoch==self.config['epochs']:
                fname1 = '{}_e{}.pt'.format(self.config['outfile'],epoch)
                self.network.to('cpu')
                torch.save({
                    'epoch': epoch,
                    'integ_state_dict': self.network.state_dict(),
                    'opti_state_dict': self.opti.state_dict()
                },fname1)
                self.network.to(self.device)

        t1=time.time()
        self.logfile=open(self.config['logfile'],'a')
        self.logfile.write("Total time (hours): "+str((t1-tstart)/3600.)+"\n")
        self.logfile.close()
        
    def Execute(self):
        """
        Execute the main program
        """
        self.OpenConfigFile()
        self.SetupLogfile()
        self.PrepareDataset()
        self.CreateNetwork()
        self.InitTraining()
        self.TrainNetwork()
        self.CloseConfigFile()