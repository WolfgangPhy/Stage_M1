import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import math
from scipy.spatial.transform import Rotation as R
import copy,time

############################################################################
#
#    Construct extinction map in 2D
#
############################################################################
#

############################################################################
#
# Class for torch dataset :
#
############################################################################
class MyDataset2D(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ell, dist, K, error):
        'Initialization'
        #self.list_IDs  = np.arange(len(ell))
        self.ell = ell
        self.cosell = np.cos(self.ell*np.pi/180.)
        self.sinell = np.sin(self.ell*np.pi/180.)
        self.dist = dist # distance in kpc
        self.K = K # total Absorption
        self.error = error # error on absorption

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ell)

    def __getitem__(self, index): ### Not sure this is how I should do it
        'Generates one sample of data'
        return torch.tensor((self.cosell[index],self.sinell[index],self.dist[index])), torch.tensor((self.K[index],self.error[index]))

############################################################################
#
# Definition of the neural network model to be used :
# here a simple dense model (perceptron) with one 
# input layer of size 3, normalized (l,b,d), one hidden layer fully connected  
# and one output layer of size 1.
# Using a sigmoid activation -> network has an analitic integral
#
############################################################################
# defines the integration neural network model to be used : here a simple dense model with one 
# input layer of size 3, one hidden layer fully connected  
# and one output layer of size 1.
# Using a sigmoid activation
class Ext2D(nn.Module):
    
    def __init__(self,hidden_size):
        super(Ext2D,self).__init__()
        self.hidden_size = hidden_size
        self.lin1 = nn.Linear(3, hidden_size, bias=True)
        self.lin2 = nn.Linear(hidden_size, 1, bias=True)
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        out = self.lin1(x)
        out = self.Sigmoid(out)
        out = self.lin2(out)
        return out


############################################################################
#
# Custom analytic integral of the network ext3D to be used in MSE loss
#
############################################################################
def integral(x,net,xmin=0.,debug=0):
    # Equation 15b of Lloyd et al 2020 -> Phi_j for each neuron
    # Li_1(x) = -ln(1-x) for x \in C
    batch_size = x.size()[0]
    n = x.size()[1] #number of coordinates the last one is the distance
    xmin = x*0. + xmin

    a = -torch.log(1.+torch.exp(-1.*(net.lin1.bias.unsqueeze(1).expand(net.hidden_size,batch_size)
                                    +torch.matmul(net.lin1.weight[:,0:n-1],torch.transpose(x[:,0:n-1],0,1)))
                               - torch.matmul(net.lin1.weight[:,n-1].unsqueeze(1),torch.transpose(xmin[:,n-1].unsqueeze(1),0,1)) ) )
    b = torch.log(1.+torch.exp(-1.*(net.lin1.bias.unsqueeze(1).expand(net.hidden_size,batch_size) 
                                    +torch.matmul(net.lin1.weight[:,0:n-1],torch.transpose(x[:,0:n-1],0,1)))
                               - torch.matmul(net.lin1.weight[:,n-1].unsqueeze(1),torch.transpose(x[:,n-1].unsqueeze(1),0,1)) ) )

    phi_j = a + b
   
    # Equation 15a of Lloyd et al 2020 : alpha_1=0, beta_1=x
    # sum over al neurons of the hidden layer
    aa = net.lin2.bias * (x[:,n-1] - xmin[:,n-1])
        
    bb = torch.matmul(net.lin2.weight[0,:] , (torch.transpose((x[:,n-1]-xmin[:,n-1]).unsqueeze(1),0,1).expand(net.hidden_size,batch_size) 
        + torch.transpose(torch.div(torch.transpose(phi_j,0,1),net.lin1.weight[:,n-1]),0,1) ) ) 
    
    res = aa + bb

    return res
    
    
############################################################################
#
# Initialisation and creation functions for the network
#
############################################################################
#Initialize weights and biases for model m
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

# Create network and set optimizer
def create_net_integ(hidden_size,lr=1e-2):
    network = Ext2D(hidden_size)
    return network, optim.Adam(network.parameters(), lr=lr)

############################################################################
#
# Loss function to be used
#
############################################################################
# log likelihood loss function
# assumes label=(E,sigma) and returns <((x-label(E))/label(sigma))**2> where <.> is either mean or sum
def loglike_loss(yhat,label,reduction='sum'):
    if reduction=='sum':
        return (((yhat-label[:,0])/label[:,1])**2).sum()
    elif reduction=='mean':
        return (((yhat-label[:,0])/label[:,1])**2).mean()
    else :
        raise Exception('reduction value unkown. Should be sum or mean')


############################################################################
#
# Function to perform one training step
# does not return the full loss function to avoid memory glitches
#
############################################################################
def take_step(xb, yb):
    global lossint_total
    global lossdens_total
    global nu_ext
    global nu_dens
    
    yb =yb.float().detach()
    xb = xb.float().to(device)
    yb = yb.to(device)
    #print(xb.size())
        
    # copy xb to new tensor and sets distance to 0
    y0 = yb.clone().detach()
    y0 = y0[:,0].unsqueeze(1) * 0.
        
    # compute ANN prediction for integration
    # density estimation at each location
    dens = network(xb)
    # total extinction : xb in [-1,1] after rescaling -> pass the lower integration bound to the code as default is 0
    exthat = integral(xb,network,xmin=-1.)
        
    # compute loss function for integration network 
    # total extinction must match observed value
    lossintegral = nu_ext * loglike_loss(exthat,yb,reduction='mean')
    # density at point xb must be positive
    #print(dens.size(),y0.size())
    lossdens = nu_dens * F.mse_loss(F.relu(-1.*dens),y0,reduction='sum')
        
    # combine loss functions
    fullloss = lossdens + lossintegral
        
    # compute total loss of epoch (for monitoring)
    lossint_total=lossint_total+lossintegral.item()
    lossdens_total=lossdens_total+lossdens.item()

    # zero gradients before taking step (gradients are additive, if not set to zero then adds to the previous gradients)
    opti.zero_grad()

    # compute gradients
    fullloss.backward()
                
    # do 1 optimisation step after minibatch
    opti.step()


############################################################################
#
# Function to perform one validation step
#
############################################################################
def validation(x_val,y_val):
    global valint_total
    global valdens_total
    global nu_ext
    global nu_dens
    
    #print(x_val.size())
    #x_val.requires_grad = True
    y_val = y_val.float().detach()
    x_val = x_val.float().to(device)
    y_val = y_val.to(device)

    y0_val = y_val.clone().detach()
    y0_val = y_val[:,0].unsqueeze(1) * 0.
                
    # density estimation at each location
    dens = network(x_val)
    # total extinction
    exthat = integral(x_val,network,xmin=-1.)
        
    # compute loss function for  network : L2 norm
    # total extinction must match observed value
    lint = nu_ext * loglike_loss(exthat,y_val,reduction='mean')

    # density at point xb must be positive
    ldens = nu_dens * F.mse_loss(F.relu(-1.*dens),y0_val,reduction='sum')
                
    valdens_total += ldens.item()        
    valint_total += lint.item()        



############################################################################
#                                                                          #
#                             MAIN PROGRAM                                 #
#                                                                          #
############################################################################
if __name__ == "__main__":
    ######### MAYBE PUT THIS IN A CONFIGURATION FILE
    datafile = './fiducial_model2D.pt'
    outfile = '../test_models/fiducial_model2D_mean'
    logfile = '../test_models/fiducial_model2D_mean.log'
    lossfile = '../test_models/fiducial_model2D_mean.loss'
    valfile = '../test_models/fiducial_model2D_mean.val'
    epochs = 600000 # number of epochs
    newnet = False # set to 0 is using an pretrained network
    pretrainednetwork = "../test_models/fiducial_model2D_mean_e340000.pt"

    ######### OPEN LOGFILE
    if newnet:
        log_file = open(logfile,'w')
    else :
        log_file = open(logfile,'a')
    
    dtype = torch.float
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    log_file.write('__Python VERSION:'+sys.version)
    log_file.write('__pyTorch VERSION:'+torch.__version__+'\n')
    if torch.cuda.is_available():
        log_file.write('__Number CUDA Devices:'+str(torch.cuda.device_count())+'\n')
        log_file.write('Active CUDA Device: GPU'+str(torch.cuda.current_device())+'\n')
        log_file.write('Available devices '+str(torch.cuda.device_count())+'\n')
        log_file.write('Current cuda device '+str(torch.cuda.current_device())+'\n')  
    log_file.write('Using device: '+str(device)+'\n')
    log_file.write('Using datafile: '+datafile+'\n')
    log_file.write('Maps stored in '+outfile+'_e*.pt\n')

    ######### HANDLING DATA
    dataset = torch.load(datafile, map_location='cpu')

    #attrs = vars(dataset)
    #print(', '.join("%s: %s" % item for item in attrs.items()))

    # Limits of the dataset of our sample
    xmin=0.
    xmax=0.

    # Computes normalisation coefficients to help training, xmin and ymin both =0
    for i in range(dataset.__len__()):
        if dataset.dist[i].item()> xmax:
            xmax = dataset.dist[i].item()
            
    log_file.write('Distance nomalisation factor (xmax):'+str(xmax)+'\n')  

    for i in range(dataset.__len__()):
        dataset.dist[i] = 2.*dataset.dist[i].item()/(xmax-xmin)-1.

    # prepares the dataset used for training and validation
    size = dataset.__len__()
    training_size= int(size*0.75)
    validation_size = size-training_size
    train_dataset, val_dataset = random_split(dataset, [training_size, validation_size])

    # prepares the dataloader for minibatch
    train_loader = DataLoader(dataset=train_dataset, batch_size=10000, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=10000,shuffle=False)

    ######### HANDLING NETWORK CREATION/INITIALISATION
    dim = 2
    k = (np.log10(dataset.__len__()))**-4.33 * 16. * dataset.__len__()/(dim+2)*2.
    log_file.write('Using '+str(int(k))+' neurons in the hidden layer\n')
    hidden_size = int(k)

    network, opti = create_net_integ(hidden_size,lr=1e-3)
    network.apply(init_weights)
    if not newnet:
        # define networks
        network = Ext2D(hidden_size)

        # load checkpoint file
        checkpoint = torch.load(pretrainednetwork,map_location='cpu')

        # update networks to checkpoint state
        network.load_state_dict(checkpoint['integ_state_dict'])

        # update optimizers state
        opti.load_state_dict(checkpoint['opti_state_dict'])

        # update epoch
        epoch = checkpoint['epoch']

    network.to(device)    
    network.train()

    ######### INITIALISATION OF TRAINING
    # initialize epoch
    try:
        epoch
    except NameError:
        epoch = -1

    # Lagrange multipliers for loss calculation
    # global variables used by fullloss
    nu_ext = 1.   # total ext must match observations
    nu_dens = 1.   # density must be positive
    log_file.write('(nu_ext,nu_dens)=('+str(nu_ext)+','+str(nu_dens)+')\n\nStart training:\n')
    log_file.close()
    
    ######### START TRAINING
    tstart = time.time()
    for idx in range(epochs+1):
        #open logfile here and close it at end of epoch to make sure everything is written
        log_file = open(logfile,'a')
    
        # set start time of epoch and epoch number
        t0 = time.time()
        epoch = epoch+1

        # initialize variables at each epoch to store full losses
        lossint_total=0.
        lossdens_total=0.

        # loop over minibatches
        nbatch = 0
        for xb,yb in train_loader:
            nbatch = nbatch+1
            take_step(xb,yb)

        # add up loss function contributions
        full_loss = lossdens_total+lossint_total # /(nbatch*1.) # loss of the integral is on the mean so we need to divide by the number of batches

        # print progress for full batch
        #if epoch%10==0:
        t1 = time.time()
        log_file.write("Epoch "+str(epoch)+" -  Loss:"+str(full_loss)+" ("+str(lossint_total)+','+str(lossdens_total)+") Total time (min): "+str((t1-t0)/60.)+"\n")
 
        loss_file = open(lossfile,'a')
        loss_file.write(str(epoch)+' '+str(full_loss)+' '+str(lossint_total)+' '+str(lossdens_total)+' '+str((t1-t0)/60.)+'\n')
        loss_file.close()
        
        # compute loss on validation sample
        if epoch%50==0:
            with torch.no_grad(): # turns off gradient calculation
                # init loss variables
                valint_total=0.
                valdens_total=0.

                # loop over minibatches of validation sample
                nbatch=0
                for x_val,y_val in val_loader:
                    nbatch=nbatch+1
                    validation(x_val, y_val)
            
                val_loss = valdens_total+valint_total/(nbatch*1.)
            
                t2 = time.time()
                log_file.write("Validation Epoch "+str(epoch)+" -  Loss:"+str(val_loss)+" ("+str(valint_total)+","+str(valdens_total)+") Time val (min):"+str((t2-t1)/60.)+" Total:"+str((t2-t0)/60.)+"\n")

                val_file = open(valfile,'a')
                val_file.write(str(epoch)+' '+str(val_loss)+' '+str(valint_total)+' '+str(valdens_total)+' '+str((t2-t1)/60.)+' '+str((t2-t0)/60.)+'\n')
                val_file.close()
                

        # end of epoch close log file to make sure the information can be read about progress
        log_file.close()
        
        # save model every 50 epoch and last step
        if epoch%10000==0 or epoch==epochs:
            fname1 = '{}_e{}.pt'.format(outfile,epoch)
            network.to('cpu')
            torch.save({
                'epoch': epoch,
                'integ_state_dict': network.state_dict(),
                'opti_state_dict': opti.state_dict()
            },fname1)
            network.to(device)

    t1=time.time()
    log_file=open(logfile,'a')
    log_file.write("Total time (hours): "+str((t1-tstart)/3600.)+"\n")
    log_file.close()
