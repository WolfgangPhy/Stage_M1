import torch.nn.functional as F

class ExtinctionNeuralNetTrainer:
    """
    A class for training the extinction neural network.

    # Args:
        - `neural_network_builder (ExtinctionNeuralNetBuilder)`: Instance of the neural network builder.
        - `int_loss_function (callable)`: Loss function for extinction.
        - `dens_loss_function (callable)`: Loss function for density.
        - `int_reduction_method (str)`: Method for reducing the extinction loss.
        - `dens_reduction_method (str)`: Method for reducing the density loss.

    # Attributes:
        - `builder (ExtinctionNeuralNetBuilder)`: Instance of the neural network builder.
        - `int_loss_function (callable)`: Loss function for extinction.
        - `dens_loss_function (callable)`: Loss function for density.
        - `int_reduction_method (str)`: Method for reducing the extinction loss.
        - `dens_reduction_method (str)`: Method for reducing the density loss.
        
    # Methods:
        - `take_step(in_batch, tar_batch, lossint_total, lossdens_total, nu_ext, nu_dens)`: Performs one training step.
        - `validation(in_batch_validation_set, tar_batch_validation_set, nu_ext, nu_dens, valint_total, valdens_total)`: Performs one validation step.

    # Example:
        >>> # Example usage of ExtinctionNeuralNetTrainer
        >>> hidden_size = 64
        >>> learning_rate = 0.001
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> builder = ExtinctionNeuralNetBuilder(device, hidden_size, learning_rate)
        >>> trainer = ExtinctionNeuralNetTrainer(builder)

        >>> # Perform one training step
        >>> in_batch = torch.randn((batch_size, 3))
        >>> tar_batch = torch.randn((batch_size, 2))  # Assuming labels (E, sigma)
        >>> nu_ext = 1.0
        >>> nu_dens = 1.0
        >>> lossint_total, lossdens_total = trainer.take_step(in_batch, tar_batch, 0.0, 0.0, nu_ext, nu_dens)

        >>> # Perform one validation step
        >>> in_batch_val = torch.randn((val_batch_size, 3))
        >>> tar_batch_val = torch.randn((val_batch_size, 2))  # Assuming validation labels (E, sigma)
        >>> valint_total, valdens_total = trainer.validation(in_batch_val, tar_batch_val, nu_ext, nu_dens, 0.0, 0.0)
    """
    
    def __init__(self, neural_network_builder, int_loss_function, dens_loss_function, int_reduction_method, dens_reduction_method):
        self.builder = neural_network_builder
        self.int_loss_function = int_loss_function
        self.dens_loss_function = dens_loss_function
        self.int_reduction_method = int_reduction_method
        self.dens_reduction_method = dens_reduction_method

    def take_step(self, in_batch, tar_batch, lossint_total, lossdens_total, nu_ext, nu_dens):
        """
        Function to perform one training step.

        This function executes one training step for the neural network model.
        It updates the model's parameters based on the provided input (in_batch) and target (tar_batch) batches.
        The loss function is a combination of log likelihood loss for total extinction and
        mean squared error loss for density estimation.

        # Args:
            - `in_batch (torch.Tensor)`: Input batch for the neural network.
            - `tar_batch (torch.Tensor)`: Target batch for the neural network.
            - `lossint_total (float)`: Total loss for extinction.
            - `lossdens_total (float)`: Total loss for density.
            - `nu_ext (float)`: Lagrange multiplier for extinction loss calculation.
            - `nu_dens (float)`: Lagrange multiplier for density loss calculation.
            
        # Returns:
            `tuple[float, float]`: Total loss for extinction, Total loss for density.
        """
        
        tar_batch = tar_batch.float().detach()
        in_batch = in_batch.float().to(self.builder.device)
        tar_batch = tar_batch.to(self.builder.device)
            
        # copy in_batch to new tensor and sets distance to 0
        y0 = tar_batch.clone().detach()
        y0 = y0[:,0].unsqueeze(1) * 0.
            
        # compute ANN prediction for integration
        # density estimation at each location
        dens = self.builder.network.forward(in_batch)
        # total extinction : in_batch in [-1,1] after rescaling -> pass the lower integration bound to the code as default is 0
        exthat = self.builder.integral(in_batch, self.builder.network, min_distance=-1.)
            
        # compute loss function for integration network 
        # total extinction must match observed value
        lossintegral = nu_ext * self.int_loss_function(exthat,tar_batch,reduction_method=self.int_reduction_method)
        
        # density at point in_batch must be positive
        lossdens = nu_dens * self.dens_loss_function(F.relu(-1.*dens),y0,reduction=self.dens_reduction_method)
            
        # combine loss functions
        fullloss = lossdens + lossintegral
        
            
        # compute total loss of epoch (for monitoring)
        lossint_total += lossintegral.item()
        lossdens_total += lossdens.item()

        # zero gradients before taking step (gradients are additive, if not set to zero then adds to the previous gradients)
        self.builder.opti.zero_grad()

        # compute gradients
        fullloss.backward()
                    
        # do 1 optimisation step after minibatch
        self.builder.opti.step()
        
        return lossint_total, lossdens_total

    def validation(self, in_batch_validation_set, tar_batch_validation_set, nu_ext, nu_dens, valint_total, valdens_total):
        """
        Function to perform one validation step.

        This function executes one validation step for the neural network model.
        It evaluates the model's performance on the validation set based on the provided input (in_batch_validation_set) and target (tar_batch_validation_set) batches.
        The loss function is a combination of log likelihood loss for total extinction and
        mean squared error loss for density estimation.

        # Args:
            - `in_batch_validation_set (torch.Tensor)`: Input batch for the validation set.
            - `tar_batch_validation_set (torch.Tensor)`: Target batch for the validation set.
            - `nu_ext (float)`: Lagrange multiplier for extinction loss calculation.
            - `nu_dens (float)`: Lagrange multiplier for density loss calculation.
            - `valint_total (float)`: Total loss for extinction in the validation set.
            - `valdens_total (float)`: Total loss for density in the validation set.
            
        # Returns:
            `tuple[float, float]`: Total loss for extinction in the validation set, Total loss for density in the validation set.
        """
        tar_batch_validation_set = tar_batch_validation_set.float().detach()
        in_batch_validation_set = in_batch_validation_set.float().to(self.builder.device)
        tar_batch_validation_set = tar_batch_validation_set.to(self.builder.device)

        y0_val = tar_batch_validation_set.clone().detach()
        y0_val = tar_batch_validation_set[:,0].unsqueeze(1) * 0.
                    
        # density estimation at each location
        dens = self.builder.network.forward(in_batch_validation_set)
        
        # total extinction
        exthat = self.builder.integral(in_batch_validation_set, self.builder.network, min_distance=-1.)
            
        # compute loss function for  network : L2 norm
        # total extinction must match observed value
        lint = nu_ext * self.int_loss_function(exthat, tar_batch_validation_set, reduction_method=self.int_reduction_method)

        # density at point in_batch must be positive
        ldens = nu_dens * self.dens_loss_function(F.relu(-1.*dens), y0_val, reduction=self.dens_reduction_method)
                    
        valdens_total += ldens.item()        
        valint_total += lint.item() 
        
        return valint_total, valdens_total  
        