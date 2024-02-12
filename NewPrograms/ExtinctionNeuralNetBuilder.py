import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ExtinctionNeuralNet as neuralnet

class ExtinctionNeuralNetBuilder:
    """
    A utility class for building and training neural networks for extinction and density estimation.

    # Args:
        `device (torch.device)`: Device on which the neural network is running.
        `hidden_size (int)`: Size of the hidden layer in the neural network.
        `learning_rate (float, optional)`: Learning rate for the Adam optimizer. Defaults to 0.001.

    # Attributes:
        `network (ExtinctionNeuralNet)`: The neural network model for extinction and density estimation.
        `opti (optim.Adam)`: The Adam optimizer used for training.

    # Methods:
        - `integral(tensor, network_model, xmin=0., debug=0)`: Custom analytic integral of the network for MSE loss.
        - `init_weights(model)`: Initializes weights and biases using Xavier uniform initialization.
        - `create_net_integ(hidden_size, learning_rate=1e-2)`: Creates a neural network and sets up the optimizer.
        - `loglike_loss(prediction, label, reduction_method='sum')`: Computes the log likelihood loss.
        - `take_step(in_batch, tar_batch)`: Performs one training step.
        - `validation(in_batch_validation_set, tar_batch_validation_set)`: Performs one validation step.
        
    # Example:
        >>> # Example usage of ExtinctionNeuralNetBuilder
        >>> hidden_size = 64
        >>> learning_rate = 0.001

        >>> # Create an instance of ExtinctionNeuralNetBuilder
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> builder = ExtinctionNeuralNetBuilder(device, hidden_size, learning_rate)

        >>> # Perform one training step
        >>> in_batch = torch.randn((batch_size, 3))
        >>> tar_batch = torch.randn((batch_size, 2))  # Assuming labels (E, sigma)
        >>> builder.take_step(in_batch, tar_batch)

        >>> # Perform one validation step
        >>> in_batch_val = torch.randn((val_batch_size, 3))
        >>> tar_batch_val = torch.randn((val_batch_size, 2))  # Assuming validation labels (E, sigma)
        >>> builder.validation(in_batch_val, tar_batch_val)
    """
    
    def __init__(self, device, hidden_size, learning_rate=0.001):
        """
        Initializes an instance of the ExtinctionNeuralNetBuilder class.

        # Args:
            `device (torch.device)`: Device on which the neural network is running.
            `hidden_size (int)`: Size of the hidden layer in the neural network.
            `learning_rate (float, optional)`: Learning rate for the Adam optimizer. Defaults to 0.001.
        """
        self.device = device
        self.network, self.opti = self.create_net_integ(hidden_size, learning_rate)
        
    def integral(tensor, network_model, xmin=0., debug=0):
        """
        Custom analytic integral of the network ext3D to be used in MSE loss.

        This function calculates a custom analytic integral of the network ext3D,
        as specified in the Equation 15a and Equation 15b of Lloyd et al. 2020,
        to be used in Mean Squared Error (MSE) loss during training.

        # Args:
            `tensor (torch.Tensor)`: Input tensor of size (batch_size, 3).
            `network_model (Ext3D)`: The neural network model (Ext3D) used for the integration.
            `xmin (float, optional)`: Minimum value for integration. Defaults to 0.
            `debug (int, optional)`: Debugging flag. Defaults to 0.

        # Returns:
            `torch.tensor`: Result of the custom analytic integral for each sample in the batch.
        """
        # Equation 15b of Lloyd et al 2020 -> Phi_j for each neuron
        # Li_1(x) = -ln(1-x) for x \in C
        batch_size = tensor.size()[0]
        coord_num = tensor.size()[1]  # number of coordinates, the last one is the distance
        xmin = tensor * 0. + xmin

        a = -torch.log(1. + torch.exp(-1. * (network_model.lin1.bias.unsqueeze(1).expand(network_model.hidden_size, batch_size)
                                                + torch.matmul(network_model.lin1.weight[:, 0:coord_num - 1], 
                                                torch.transpose(tensor[:, 0:coord_num - 1], 0, 1))\
                                            )
                                        - torch.matmul(network_model.lin1.weight[:, coord_num - 1].unsqueeze(1), 
                                                            torch.transpose(xmin[:, coord_num - 1].unsqueeze(1), 0, 1)
                                                        )
                                    )
                    )
        b = torch.log(1. + torch.exp(-1. * (network_model.lin1.bias.unsqueeze(1).expand(network_model.hidden_size, batch_size)
                                                + torch.matmul(network_model.lin1.weight[:, 0:coord_num - 1], 
                                                torch.transpose(tensor[:, 0:coord_num - 1], 0, 1))
                                            )
                                        - torch.matmul(network_model.lin1.weight[:, coord_num - 1].unsqueeze(1),
                                                            torch.transpose(tensor[:, coord_num - 1].unsqueeze(1), 0, 1)
                                                            )
                                        )
                        )

        phi_j = a + b

        # Equation 15a of Lloyd et al 2020: alpha_1=0, beta_1=x
        # Sum over all neurons of the hidden layer
        aa = network_model.lin2.bias * (tensor[:, coord_num - 1] - xmin[:, coord_num - 1])

        bb = torch.matmul(network_model.lin2.weight[0, :],
                            (torch.transpose( (tensor[:, coord_num - 1] - xmin[:, coord_num - 1])
                                            .unsqueeze(1), 0, 1)
                                .expand(network_model.hidden_size, batch_size)
                                + torch.transpose(torch.div(torch.transpose(phi_j, 0, 1), 
                                                            network_model.lin1.weight[:, coord_num - 1]),
                                                0, 1)
                                )
                            )

        result = aa + bb

        return result

    def init_weights(model):
        """
        Function to initialize weights and biases for the given PyTorch model.

        This function initializes the weights and biases of the linear layers in the model
        using Xavier (Glorot) uniform initialization for weights and sets bias values to 0.1.

        # Args:
            `model (torch.nn.Module)`: The PyTorch model for which weights and biases need to be initialized.
        """
        if type(model) == nn.Linear:
            torch.nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.1)
            
    def create_net_integ(hidden_size, learning_rate=1e-2):
        """
        Function to create the neural network and set the optimizer.

        This function creates an instance of the ExtinctionNeuralNet neural network with the specified
        hidden size and initializes an Adam optimizer with the given learning rate.

        # Args:
            `hidden_size (int)`: Size of the hidden layer in the neural network.
            `learning_rate (float, optional)`: Learning rate for the Adam optimizer. Defaults to 1e-2.

        # Returns:
            `tuple[ExtinctionNeuralNet, optim.Adam]`: A tuple containing the created neural network and the Adam optimizer.
        """
        network = neuralnet.ExtinctionNeuralNet(hidden_size)
        return network, optim.Adam(network.parameters(), lr=learning_rate)

    def loglike_loss(prediction, label, reduction_method='sum'):
        """
        Function to compute the log likelihood loss.

        This function implements the log likelihood loss function. It assumes that 'label' is a pair (E, sigma)
        and returns <((x - label(E)) / label(sigma))**2>, where <.> is either the mean or the sum, depending on the 'reduction' parameter.

        # Args:
            `prediction (torch.Tensor)`: Model predictions.
            `label (torch.Tensor)`: Labels in the form (E, sigma).
            `reduction_method (str, optional)`: Method for reducing the loss, 'sum' by default.

        # Raises:
            `Exception`: Raised if the reduction value is unknown. Should be 'sum' or 'mean'.

        # Returns:
            `torch.Tensor`: Value of the log likelihood loss.
        """
        if reduction_method=='sum':
            return ( ( ( prediction-label[:,0] )/label[:,1] )**2 ).sum()
        elif reduction_method=='mean':
            return ( ( ( prediction-label[:,0] )/label[:,1] )**2 ).mean()
        else :
            raise Exception('reduction value unkown. Should be sum or mean')

    def take_step(self, in_batch, tar_batch, lossint_total, lossdens_total, nu_ext, nu_dens):
        """
        Function to perform one training step.

        This function executes one training step for the neural network model.
        It updates the model's parameters based on the provided input (in_batch) and target (tar_batch) batches.
        The loss function is a combination of log likelihood loss for total extinction and
        mean squared error loss for density estimation.

        # Args:
            `in_batch (torch.Tensor)`: Input batch for the neural network.
            `tar_batch (torch.Tensor)`: Target batch for the neural network.
            `lossint_total (float)`: Total loss for extinction.
            `lossdens_total (float)`: Total loss for density.
            `nu_ext (float)`: Coefficient for extinction loss.
            `nu_dens (float)`: Coefficient for density loss.
        """
        
        tar_batch = tar_batch.float().detach()
        in_batch = in_batch.float().to(self.device)
        tar_batch = tar_batch.to(self.device)
        #print(in_batch.size())
            
        # copy in_batch to new tensor and sets distance to 0
        y0 = tar_batch.clone().detach()
        y0 = y0[:,0].unsqueeze(1) * 0.
            
        # compute ANN prediction for integration
        # density estimation at each location
        dens = self.network(in_batch)
        # total extinction : in_batch in [-1,1] after rescaling -> pass the lower integration bound to the code as default is 0
        exthat = ExtinctionNeuralNetBuilder.integral(in_batch, self.network, xmin=-1.)
            
        # compute loss function for integration network 
        # total extinction must match observed value
        lossintegral = nu_ext * ExtinctionNeuralNetBuilder.loglike_loss(exthat,tar_batch,reduction_method='mean')
        # density at point in_batch must be positive
        #print(dens.size(),y0.size())
        lossdens = nu_dens * F.mse_loss(F.relu(-1.*dens),y0,reduction='sum')
            
        # combine loss functions
        fullloss = lossdens + lossintegral
            
        # compute total loss of epoch (for monitoring)
        lossint_total = lossint_total+lossintegral.item()
        lossdens_total = lossdens_total+lossdens.item()

        # zero gradients before taking step (gradients are additive, if not set to zero then adds to the previous gradients)
        self.opti.zero_grad()

        # compute gradients
        fullloss.backward()
                    
        # do 1 optimisation step after minibatch
        self.opti.step()

    def validation(self, in_batch_validation_set, tar_batch_validation_set):
        """
        Function to perform one validation step.

        This function executes one validation step for the neural network model.
        It evaluates the model's performance on the validation set based on the provided input (in_batch_validation_set) and target (tar_batch_validation_set) batches.
        The loss function is a combination of log likelihood loss for total extinction and
        mean squared error loss for density estimation.

        # Args:
            `in_batch_validation_set (torch.Tensor)`: Input batch for the validation set.
            `tar_batch_validation_set (torch.Tensor)`: Target batch for the validation set.
        """
        global valint_total
        global valdens_total
        global nu_ext
        global nu_dens
        
        #print(in_batch_validation_set.size())
        #in_batch_validation_set.requires_grad = True
        tar_batch_validation_set = tar_batch_validation_set.float().detach()
        in_batch_validation_set = in_batch_validation_set.float().to(self.device)
        tar_batch_validation_set = tar_batch_validation_set.to(self.device)

        y0_val = tar_batch_validation_set.clone().detach()
        y0_val = tar_batch_validation_set[:,0].unsqueeze(1) * 0.
                    
        # density estimation at each location
        dens = self.network(in_batch_validation_set)
        # total extinction
        exthat = ExtinctionNeuralNetBuilder.integral(in_batch_validation_set, self.network, xmin=-1.)
            
        # compute loss function for  network : L2 norm
        # total extinction must match observed value
        lint = nu_ext * ExtinctionNeuralNetBuilder.loglike_loss(exthat,tar_batch_validation_set,reduction_method='mean')

        # density at point in_batch must be positive
        ldens = nu_dens * F.mse_loss(F.relu(-1.*dens),y0_val,reduction='sum')
                    
        valdens_total += ldens.item()        
        valint_total += lint.item()   
        